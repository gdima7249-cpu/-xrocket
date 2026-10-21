"""Можно ли угадывать направление РЕАЛЬНЫХ валютных пар Deriv (frxGBPCAD, frxEURUSD ...) на 15/30/60 минут лучше порога выплаты?

Ставки Rise/Fall на валюты Deriv даёт от 15 минут. Скрипт:
  1) качает 5-минутные свечи пары из публичной ленты Deriv (ключ не нужен);
  2) walk-forward (учимся на прошлом, проверяем на будущем): модель по последним свечам и правило отката
     «резкое падение (в единицах обычного разброса) → вверх»; доля угаданных против порога безубыточности;
  3) с флагом --payout спрашивает у Deriv (демо-счёт из .env) реальную выплату на этих сроках. Ничего не покупает.

  python deriv_forex_test.py --symbol frxGBPCAD
  python deriv_forex_test.py --symbol frxEURUSD --payout
  python deriv_forex_test.py --synthetic              # случайные цены: везде должно быть около 50%
"""
import argparse
import json
import time

import numpy as np

import deriv_ticks_test as T

LAGS = 12


def fetch_candles(symbol, total, gran=300, ws=None, pause=2.0):
    """5-минутные свечи (по 5000 за запрос, назад во времени), без дублей, по возрастанию времени: (время, close).
    Deriv ограничивает частоту таких запросов: между запросами пауза, при упоре в лимит — ожидание и повтор.
    Скачанное сохраняется в research_cache (при повторном запуске в течение суток берётся оттуда)."""
    import os
    os.makedirs("research_cache", exist_ok=True)
    path = f"research_cache/DERIV_{symbol}_{gran}s_{total}.json"
    if ws is None and os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        print(f"Свечи {symbol} взяты из кэша: {len(d['t'])}")
        return np.array(d["t"]), np.array(d["c"])
    own = ws is None
    if own:
        import websocket
        ws = websocket.create_connection(T.WS_URL, timeout=30)
    seen, end, req, stalled, waits = {}, "latest", 0, 0, 0
    try:
        while len(seen) < total and req < 200 and stalled < 3:
            ws.send(json.dumps({"ticks_history": symbol, "style": "candles", "granularity": gran, "count": 5000,
                                "end": end, "adjust_start_time": 1}))
            req += 1
            msg = json.loads(ws.recv())
            if msg.get("error"):
                text = str(msg["error"].get("message", ""))
                if "rate limit" in text.lower() and waits < 10:
                    waits += 1
                    wait = 30 * waits
                    print(f"  Deriv просит подождать (лимит запросов), жду {wait} с... (уже скачано {len(seen)} свечей)")
                    time.sleep(wait)
                    continue
                raise SystemExit(f"Deriv API: {text} (проверьте символ, например frxGBPCAD)")
            cs = msg.get("candles") or []
            if not cs:
                break
            before = len(seen)
            for c in cs:
                seen.setdefault(int(c["epoch"]), float(c["close"]))
            stalled = stalled + 1 if len(seen) - before < len(cs) * 0.5 else 0
            end = min(int(c["epoch"]) for c in cs) - 1
            print(f"  скачано {len(seen)} свечей...")
            time.sleep(pause)
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass
    ts = sorted(seen)
    if not ts:
        raise SystemExit("Свечей нет")
    print(f"Свечей: {len(ts)} (запросов {req}); период {time.strftime('%Y-%m-%d', time.gmtime(ts[0]))} … "
          f"{time.strftime('%Y-%m-%d', time.gmtime(ts[-1]))}")
    if own:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"t": ts, "c": [seen[t] for t in ts]}, f)
    return np.array(ts), np.array([seen[t] for t in ts])


def features(c, h):
    """X — последние LAGS изменений (в долях обычного разброса) и их суммы; y — цена через h свечей выше текущей (1) или ниже (0).
    Свечи, у которых следующие h свечей пересекают перерыв в торговле (выходные), отбрасываются."""
    r = np.diff(np.log(c))
    vol = np.std(r) + 1e-12
    X, Y, Z = [], [], []
    for t in range(LAGS + 48, len(r) - h):
        w = r[t - LAGS:t] / vol
        fut = c[t + h] - c[t]
        if fut == 0:
            continue
        X.append(list(w) + [w[-3:].sum(), w[-6:].sum(), w.sum(), r[t - 48:t].std() / vol])
        Y.append(1.0 if fut > 0 else 0.0)
        Z.append(w[-3:].sum())                                      # изменение за последние 15 минут (в разбросах)
    return np.array(X), np.array(Y), np.array(Z)


def walk(X, Y, Z, h, folds=5):
    n = len(Y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    res = {"модель, все": [], "модель, уверенные 20%": [], "откат: падение >3 разбросов за 15м → вверх": [], "всегда вверх": []}
    for k in range(1, folds):
        tr, te = slice(0, edges[k] - h), slice(edges[k], edges[k + 1])
        m = T.HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=5.0,
                                             min_samples_leaf=200, early_stopping=False, random_state=0).fit(X[tr], Y[tr])
        p = m.predict_proba(X[te])[:, 1]
        thr = np.quantile(np.abs(m.predict_proba(X[tr])[:, 1] - 0.5), 0.8)
        y = Y[te]
        side = (p > 0.5).astype(float)
        conf = np.abs(p - 0.5) >= thr
        res["модель, все"].append(((side == y).sum(), len(y)))
        res["модель, уверенные 20%"].append(((side[conf] == y[conf]).sum(), int(conf.sum())))
        dip = Z[te] <= -3 * np.sqrt(3)                              # падение больше 3 «обычных разбросов» за 15 мин
        res["откат: падение >3 разбросов за 15м → вверх"].append((int(y[dip].sum()), int(dip.sum())))
        res["всегда вверх"].append((int(y.sum()), len(y)))
    return res


def payouts(symbol, minutes):
    import deriv_demo_bot as B
    tr = B.Trader(B.connect(), 1.0, 1, 10 ** 9, 10 ** 9, unit="m")
    out = {}
    for mnt in minutes:
        try:
            pr = tr.c.call({"proposal": 1, "amount": 10, "basis": "stake", "contract_type": "CALL", "currency": tr.currency,
                            "duration": mnt, "duration_unit": "m", "underlying_symbol": symbol})["proposal"]
            out[mnt] = float(pr["payout"]) / float(pr["ask_price"]) - 1
        except Exception as e:  # noqa: BLE001
            print(f"  выплата на {mnt} мин: {str(e)[:120]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="frxGBPCAD")
    ap.add_argument("--candles", type=int, default=60000, help="сколько 5-минутных свечей скачать (60000 ≈ 8 месяцев торговли)")
    ap.add_argument("--payout", action="store_true", help="спросить у Deriv реальную выплату (нужен .env демо-счёта)")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.synthetic:
        c = T.synthetic(a.candles)
    else:
        _, c = fetch_candles(a.symbol, a.candles)
    if len(c) < 5000:
        raise SystemExit("Слишком мало свечей для проверки")
    pays = payouts(a.symbol, [15, 30, 60]) if a.payout else {}
    print(f"\n{a.symbol}: {len(c)} свечей. Доля угаданных на будущих кусках истории (4 куска).")
    for h, mnt in ((3, 15), (6, 30), (12, 60)):
        X, Y, Z = features(c, h)
        res = walk(X, Y, Z, h)
        pay = pays.get(mnt)
        be = 1 / (1 + pay) if pay else None
        head = f"\nСрок {mnt} мин" + (f": выплата Deriv {100 * pay:.0f}%, порог безубыточности {100 * be:.1f}%" if be else
                                       " (выплата не запрошена; порог при 85% — 54.1%, при 80% — 55.6%)")
        print(head)
        for name, parts in res.items():
            n = sum(m for _, m in parts)
            w = sum(int(v) for v, _ in parts)
            if not n:
                print(f"  {name:<46} нет случаев")
                continue
            p = w / n
            ci = 1.96 * np.sqrt(p * (1 - p) / n)
            per = " ".join(f"{100 * v / m:.1f}" for v, m in parts if m)
            mark = ""
            if be and p - ci > be:
                mark = "  ← выше порога с запасом"
            print(f"  {name:<46}{n:>7} ставок  {100 * p:5.1f}% ± {100 * ci:.1f}   [{per}]{mark}")
    print("\nКак читать: чтобы ставки были в плюсе, «угадано» должно быть ВЫШЕ порога безубыточности с запасом больше погрешности"
          "\n(±) и держаться во всех кусках. Окна ставок перекрываются, поэтому настоящая погрешность ещё больше напечатанной.")


if __name__ == "__main__":
    main()
