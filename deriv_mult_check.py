"""Проверка: можно ли заработать на мультипликаторах Deriv (MULTUP/MULTDOWN) по сигналу из тиков cryBTCUSD. Ничего не покупает.

Шаг 1 (нужен .env с DERIV_TOKEN, DERIV_APP_ID, DERIV_ACCOUNT_ID — демо): спрашивает у Deriv реальные условия — доступные множители,
        комиссию и уровень стоп-аута (запрос цены proposal, без покупки).
Шаг 2 (без ключа): качает историю тиков, учит ту же модель, что и тиковый бот, и на будущих кусках (walk-forward) считает,
        на сколько в среднем цена уходит в сторону сигнала за N тиков. Это сравнивается с комиссией.

Как считается прибыль мультипликатора: прибыль ≈ ставка × множитель × (изменение цены в долях) − комиссия.
Комиссия Deriv берётся с полного объёма (ставка × множитель), поэтому множитель её НЕ уменьшает: чтобы быть в плюсе, среднее движение
в сторону сигнала (в долях цены) должно быть больше комиссии в долях объёма. Именно это и сравнивает таблица.

  python deriv_mult_check.py                       # cryBTCUSD, шаги 1 и 2
  python deriv_mult_check.py --symbol cryETHUSD
  python deriv_mult_check.py --no-api               # только шаг 2 (без ключа)
  python deriv_mult_check.py --synthetic --no-api   # проверка кода без интернета
"""
import argparse

import numpy as np

import deriv_ticks_test as T

LAGS = 20


def make_xr(prices, ahead, lags=LAGS):
    """Те же признаки, что в make_xy; ответ — лог-доходность от входа (следующий тик) до выхода через ahead тиков."""
    r = np.diff(np.log(prices))
    vol = np.std(r) + 1e-12
    X, R = [], []
    for t in range(lags + 60, len(r) - ahead):
        w = r[t - lags:t] / vol
        X.append(list(w) + [w[-5:].sum(), w[-10:].sum(), w.sum(), r[t - 60:t].std() / vol])
        R.append(np.log(prices[t + 1 + ahead] / prices[t + 1]))
    return np.array(X), np.array(R)


def walk_forward_returns(X, R, top=0.2, folds=5, ahead=5):
    """Возвращает для уверенных сигналов (топ-доля) массив доходностей «в сторону сигнала» на проверочных кусках."""
    n = len(R)
    edges = np.linspace(0, n, folds + 1).astype(int)
    out, per = [], []
    for k in range(1, folds):
        tr, te = slice(0, edges[k] - ahead - 1), slice(edges[k], edges[k + 1])
        ytr = R[tr]
        ok = ytr != 0
        m = T.HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=5.0,
                                             min_samples_leaf=200, early_stopping=False, random_state=0)
        m.fit(X[tr][ok], (ytr[ok] > 0).astype(float))
        thr = np.quantile(np.abs(m.predict_proba(X[tr])[:, 1] - 0.5), 1 - top)
        p = m.predict_proba(X[te])[:, 1]
        conf = np.abs(p - 0.5) >= thr
        signed = np.where(p > 0.5, 1, -1)[conf] * R[te][conf]
        # сделки не должны перекрываться: берём каждый сигнал не чаще, чем раз в ahead тиков
        idx = np.flatnonzero(conf)
        keep, last = [], -10 ** 9
        for j, i in enumerate(idx):
            if i - last > ahead:
                keep.append(j)
                last = i
        s = signed[keep]
        out.append(s)
        per.append(s.mean() if len(s) else float("nan"))
    return np.concatenate(out) if out else np.array([]), per


def api_conditions(symbol, stake):
    """Шаг 1: условия мультипликаторов у Deriv (демо-счёт). Возвращает {множитель: комиссия в долях объёма}."""
    import deriv_demo_bot as B
    c = B.connect()
    tr = B.Trader(c, stake, 1, 10 ** 9, 10 ** 9, unit="t")
    try:
        cf = c.call({"contracts_for": symbol})
    except Exception:  # noqa: BLE001
        cf = c.call({"contracts_for": 1, "underlying_symbol": symbol})
    av = cf.get("contracts_for", cf).get("available", [])
    mults = set()
    for a in av:
        if a.get("contract_type") in ("MULTUP", "MULTDOWN"):
            mults.update(int(x) for x in a.get("multiplier_range", []) or [])
    mults = sorted(mults) or [10, 20, 50, 100]
    print(f"\nШаг 1. Условия Deriv для {symbol} (ставка {stake} {tr.currency}, ничего не покупается). Множители: {mults}")
    print(f"  {'множитель':>10}{'объём':>10}{'комиссия':>11}{'комиссия, % объёма':>21}{'стоп-аут при движении против':>31}")
    res = {}
    for m in mults:
        try:
            pr = c.call({"proposal": 1, "amount": stake, "basis": "stake", "contract_type": "MULTUP", "currency": tr.currency,
                         "multiplier": m, "underlying_symbol": symbol})["proposal"]
        except Exception as e:  # noqa: BLE001
            print(f"  {m:>10}  ошибка: {str(e)[:120]}")
            continue
        com = float(pr.get("commission") or 0)
        so = (pr.get("limit_order") or {}).get("stop_out") or {}
        so_amt = so.get("order_amount")
        rate = com / (stake * m)
        res[m] = rate
        so_txt = f"{100 * abs(float(so_amt)) / (stake * m):.2f}% (убыток {so_amt})" if so_amt is not None else "?"
        print(f"  {m:>10}{stake * m:>10.0f}{com:>11.4f}{100 * rate:>20.4f}%{so_txt:>31}")
        if com == 0:
            print("     комиссия 0 — возможно, поле называется иначе. Ответ Deriv:", {k: pr[k] for k in list(pr)[:12]})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="cryBTCUSD")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--ticks", type=int, default=100000)
    ap.add_argument("--horizons", default="5,10,20,60,120")
    ap.add_argument("--top", type=float, default=0.2)
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    rates = {} if a.no_api else api_conditions(a.symbol, a.stake)
    com = min(rates.values()) if rates else None

    prices = T.synthetic(a.ticks) if a.synthetic else T.fetch_ticks(a.symbol, a.ticks)
    print(f"\nШаг 2. История: {len(prices)} тиков {a.symbol}. Сигнал — модель, топ-{100 * a.top:.0f}% уверенных, сделки не перекрываются.")
    print("  Движение в сторону сигнала — в базисных пунктах (1 б.п. = 0.01% цены).")
    hdr = f"  {'тиков':>6}{'сделок':>8}{'угадано':>9}{'среднее, б.п.':>15}{'± погр.':>9}{'по кускам':>28}"
    if com is not None:
        hdr += f"{'комиссия, б.п.':>16}{'итог на сделку':>16}"
    print(hdr)
    for h in [int(x) for x in a.horizons.split(",")]:
        X, R = make_xr(prices, h)
        s, per = walk_forward_returns(X, R, a.top, ahead=h)
        if not len(s):
            continue
        se = s.std() / np.sqrt(len(s))
        row = (f"  {h:>6}{len(s):>8}{100 * (s > 0).mean():>8.1f}%{1e4 * s.mean():>15.3f}{1e4 * 2 * se:>9.2f}"
               f"{' '.join(f'{1e4 * v:.2f}' for v in per):>28}")
        if com is not None:
            net = s.mean() - com
            row += f"{1e4 * com:>16.3f}{1e4 * net:>15.3f}" + ("  ← плюс выше погрешности" if net > 2 * se else "")
        print(row)
    print("\nКак читать: «итог на сделку» > 0 значит, что движение в сторону сигнала в среднем больше комиссии (после неё — плюс).")
    if com is None:
        print("Комиссию не спрашивали (--no-api). Сравните «среднее, б.п.» с комиссией из шага 1: нужно среднее БОЛЬШЕ комиссии.")
    print("Оговорки: не учтены задержка исполнения (вход может быть не на следующем тике), спред и проскальзывание при закрытии;"
          " на живом счёте результат обычно хуже, чем на истории. «± погр.» — 95%-погрешность среднего: если среднее (или итог) меньше неё, это может быть просто шум.")


if __name__ == "__main__":
    main()
