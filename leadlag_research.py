"""BTC тянет альткоины: если BTC резко сдвинулся за 5 минут, а альткоин отстал — догонит ли он (и хватит ли этого на комиссию)?

Данные: 5-минутные свечи Binance (те же, что у rebound_research.py; если они уже скачаны, берутся из research_cache). Ничего не покупает.

Правило (без подглядывания в будущее):
  событие  — BTC за последнюю 5-минутную свечу изменился не меньше чем на X% (вверх или вниз),
             а альткоин за ту же свечу прошёл в ту же сторону меньше половины движения BTC («отстал»);
  вход     — альткоин по цене ОТКРЫТИЯ следующей свечи, в сторону движения BTC (BTC вверх — покупка, вниз — шорт);
  выход    — по закрытию через H свечей (5 мин / 15 мин / 1 ч);
  издержки — 0.04% комиссии + 0.02% проскальзывания за сторону (меняется --fee/--slip).
Проверки: погрешность по дням, первая и вторая половина периода, случайный вход в ту же сторону для сравнения.

  python leadlag_research.py                      # 365 дней, альткоины из rebound_research
  python leadlag_research.py --slip 0.001
  python leadlag_research.py --synthetic          # случайные независимые цены: эффекта быть не должно
  python leadlag_research.py --synthetic --lag    # искусственное запаздывание: эффект ДОЛЖЕН найтись (проверка, что метод его видит)
"""
import argparse

import numpy as np
import pandas as pd

import rebound_research as RR


def synthetic_pair(days, n_alts, lag):
    """BTC и альткоины на случайных ценах. При lag=True альткоин повторяет 60% движения BTC с опозданием на одну свечу."""
    rng = np.random.default_rng(7)
    n = days * 288
    btc_r = rng.standard_t(3, n) * 0.0015
    out = {"BTC": btc_r}
    for j in range(n_alts):
        own = rng.standard_t(3, n) * 0.002
        follow = np.r_[0, btc_r[:-1]] * 0.6 if lag else 0.0
        out[f"ALT{j}"] = own + follow
    t0 = (1_700_000_000_000 // RR.BAR_MS) * RR.BAR_MS
    dfs = {}
    for k, r in out.items():
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]]
        dfs[k] = pd.DataFrame({"time": t0 + np.arange(n) * float(RR.BAR_MS), "open": open_, "close": close})
    return dfs


def trades(btc, alt, thr, hold):
    """Сделки по одной паре. Возвращает (время входа, доходность без издержек, сторона)."""
    m = pd.merge(btc[["time", "close"]], alt[["time", "open", "close"]], on="time", suffixes=("_b", ""))
    cb, o, c, t = m["close_b"].to_numpy(), m["open"].to_numpy(), m["close"].to_numpy(), m["time"].to_numpy()
    rb = np.r_[np.nan, cb[1:] / cb[:-1] - 1]
    ra = np.r_[np.nan, c[1:] / c[:-1] - 1]
    n = len(m)
    T, R = [], []
    i = 1
    while i < n - hold - 1:
        side = 1 if rb[i] >= thr else (-1 if rb[i] <= -thr else 0)
        if side and side * ra[i] < 0.5 * abs(rb[i]):
            T.append(t[i + 1])
            R.append(side * (c[i + hold] / o[i + 1] - 1))
            i += hold
        else:
            i += 1
    return np.array(T), np.array(R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alts", default="ETH,SOL,XRP,DOGE,BNB,ADA,AVAX,LINK,LTC")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lag", action="store_true", help="с --synthetic: встроить запаздывание альткоинов")
    a = ap.parse_args()
    cost = 2 * (a.fee + a.slip)

    if a.synthetic:
        data = synthetic_pair(a.days, len(a.alts.split(",")), a.lag)
        btc = data.pop("BTC")
    else:
        btc = RR.download_5m("BTCUSDT", a.days)
        data = {}
        for s in a.alts.split(","):
            try:
                data[s] = RR.download_5m(s + "USDT", a.days)
            except Exception as e:  # noqa: BLE001
                print(f"{s}: пропуск ({type(e).__name__}: {str(e)[:80]})")
    mid = float(np.median(btc["time"]))
    print(f"BTC: {len(btc)} свечей, альткоинов: {len(data)}. Издержки на сделку: {100 * cost:.2f}%.")
    print("Доходности в % на сделку ПОСЛЕ издержек; «случ.» — случайный вход в альткоин на тот же срок (после издержек, без стороны).\n")
    print(f"  {'BTC за 5 мин':<16}{'выход':>7}{'сделок':>8}{'дней':>6}{'угадано':>9}{'в среднем':>11}{'случ.':>8}"
          f"{'t (дни)':>9}{'1-я пол.':>10}{'2-я пол.':>10}")
    rows = 0
    good = 0
    for thr in (0.005, 0.01, 0.015, 0.02):
        for hold, hname in ((1, "5м"), (3, "15м"), (12, "1ч")):
            T, R = [], []
            for df in data.values():
                t, r = trades(btc, df, thr, hold)
                T.append(t)
                R.append(r)
            t, r = np.concatenate(T), np.concatenate(R)
            if len(r) < 20:
                continue
            rows += 1
            net = r - cost
            tt, nd = RR.day_t(t, net)
            h1, h2 = net[t < mid], net[t >= mid]
            ok = net.mean() > 0 and len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0 and tt > 3
            good += bool(ok)
            print(f"  {'≥' + format(100 * thr, '.1f') + '%':<16}{hname:>7}{len(net):>8}{nd:>6}{100 * (r > 0).mean():>8.1f}%"
                  f"{100 * net.mean():>10.3f}%{-100 * cost:>7.3f}%{tt:>9.1f}"
                  f"{100 * h1.mean() if len(h1) else float('nan'):>9.3f}%{100 * h2.mean() if len(h2) else float('nan'):>9.3f}%"
                  + ("  ← устойчивый плюс" if ok else ""))
    print(f"\nПроверено вариантов: {rows}. Устойчивых (плюс после издержек в обеих половинах и t > 3): {good}.")
    print("Как читать: «угадано» выше 50% без плюса после издержек значит, что альткоин догоняет, но слишком мало, чтобы покрыть комиссию.")


if __name__ == "__main__":
    main()
