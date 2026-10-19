"""Откат после резких движений: если крипта резко упала (или выросла) за 5–30 минут, идёт ли она потом обратно — и хватает ли этого на комиссию?

Данные: 5-минутные свечи Binance за год (публично, ключи не нужны). Ничего не покупает.

Правила (без подглядывания в будущее):
  событие   — изменение цены за последние k свечей (k=1/3/6 → 5/15/30 минут) не меньше порога X% вниз (или вверх);
  вход      — по цене ОТКРЫТИЯ следующей свечи; сторона — против движения (после падения покупка, после роста — шорт);
  выход     — по цене закрытия через H свечей (15 мин / 1 ч / 4 ч);
  издержки  — комиссия фьючерсов 0.04% за вход и 0.04% за выход + проскальзывание (по умолчанию 0.02% за сторону);
  сделки по одной монете не перекрываются.

Проверки против самообмана:
  * погрешность считается по ДНЯМ (в один обвал попадают сразу многие монеты — это не независимые сделки);
  * отдельно первая и вторая половина периода — настоящий эффект должен быть в обеих;
  * сравнение со «случайным входом» в ту же сторону на тот же срок (рынок мог просто расти или падать весь год);
  * вариантов много (порог × окно × срок × сторона), поэтому лучший из них почти всегда выглядит хорошо случайно —
    верить можно только тому, что в плюсе в ОБЕИХ половинах и с t > 3.

  python rebound_research.py                         # 10 монет, 365 дней
  python rebound_research.py --days 730 --coins BTC,ETH,SOL
  python rebound_research.py --synthetic             # проверка кода без интернета (случайные цены: эффекта быть не должно)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

API = "https://data-api.binance.vision/api/v3/klines"
CACHE = "research_cache"
BAR_MS = 5 * 60 * 1000
COINS = "BTC,ETH,SOL,XRP,DOGE,BNB,ADA,AVAX,LINK,LTC"


def download_5m(symbol, days):
    import requests
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{symbol}_5m_{days}d.csv"
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
        return pd.read_csv(path)
    end = int(time.time() * 1000)
    since = end - days * 86400 * 1000
    rows = []
    while since < end:
        for attempt in range(4):
            try:
                r = requests.get(API, params={"symbol": symbol, "interval": "5m", "startTime": since, "limit": 1000}, timeout=20)
                r.raise_for_status()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        batch = r.json()
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + BAR_MS
        time.sleep(0.05)
    df = pd.DataFrame([[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows],
                      columns=["time", "open", "high", "low", "close"]).drop_duplicates("time")
    df.to_csv(path, index=False)
    return df


def synthetic_5m(days, seed):
    rng = np.random.default_rng(seed)
    n = days * 288
    ret = rng.standard_t(3, n) * 0.0015                         # «толстые хвосты», как у крипты, но без памяти
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    t0 = (int(time.time() * 1000) // BAR_MS) * BAR_MS - n * BAR_MS
    return pd.DataFrame({"time": t0 + np.arange(n) * float(BAR_MS), "open": open_, "high": np.maximum(open_, close),
                         "low": np.minimum(open_, close), "close": close})


def events(df, k, thr, hold, direction):
    """Сделки «против движения». direction=-1: после падения на thr за k свечей — покупка; +1: после роста — шорт.
    Возвращает (время входа, доходность сделки без издержек)."""
    o, c, t = df["open"].to_numpy(), df["close"].to_numpy(), df["time"].to_numpy()
    n = len(c)
    move = np.full(n, np.nan)
    move[k:] = c[k:] / c[:-k] - 1
    trig = (move <= -thr) if direction < 0 else (move >= thr)
    side = -direction                                          # против движения
    out_t, out_r = [], []
    i = k
    while i < n - hold - 1:
        if trig[i]:
            entry, exit_ = o[i + 1], c[i + hold]
            out_t.append(t[i + 1])
            out_r.append(side * (exit_ / entry - 1))
            i += hold                                          # не перекрываемся
        else:
            i += 1
    return np.array(out_t), np.array(out_r)


def baseline(df, hold, side):
    """Средняя доходность «случайного входа» в ту же сторону на тот же срок (дрейф рынка за период)."""
    c = df["close"].to_numpy()
    r = c[hold:] / c[:-hold] - 1
    return side * float(np.mean(r))


def day_t(times, net):
    """t-статистика по дням: суммируем сделки одного дня, считаем t по дневным суммам."""
    if len(net) < 2:
        return float("nan"), 0
    d = pd.Series(net).groupby((np.asarray(times) // 86_400_000).astype(np.int64)).sum()
    if len(d) < 2 or d.std() == 0:
        return float("nan"), len(d)
    return float(d.mean() / (d.std() / np.sqrt(len(d)))), len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default=COINS)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--fee", type=float, default=0.0004, help="комиссия за сторону (0.0004 = 0.04%, фьючерсы Binance, тейкер)")
    ap.add_argument("--slip", type=float, default=0.0002, help="проскальзывание за сторону")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cost = 2 * (a.fee + a.slip)

    data = {}
    for i, s in enumerate(a.coins.split(",")):
        try:
            df = synthetic_5m(a.days, i) if a.synthetic else download_5m(s + "USDT", a.days)
            if len(df) > 1000:
                data[s] = df
                print(f"{s}: {len(df)} свечей")
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {str(e)[:80]})")
    if not data:
        raise SystemExit("Нет данных")
    t_all = np.concatenate([d["time"].to_numpy() for d in data.values()])
    mid = float(np.median(t_all))

    print(f"\nИздержки на сделку: {100 * cost:.2f}% (комиссия {100 * a.fee:.2f}% + проскальзывание {100 * a.slip:.2f}% за сторону).")
    print("Доходности в % на сделку ПОСЛЕ издержек; «случ.» — случайный вход в ту же сторону на тот же срок (тоже после издержек).\n")
    print(f"  {'событие':<24}{'выход':>7}{'сделок':>8}{'дней':>6}{'угадано':>9}{'в среднем':>11}{'случ.':>8}"
          f"{'t (дни)':>9}{'1-я пол.':>10}{'2-я пол.':>10}")
    rows = []
    for direction, name in ((-1, "падение"), (1, "рост")):
        for k, kname in ((1, "5м"), (3, "15м"), (6, "30м")):
            for thr in (0.01, 0.02, 0.03, 0.05):
                for hold, hname in ((3, "15м"), (12, "1ч"), (48, "4ч")):
                    T, R, B = [], [], []
                    for s, df in data.items():
                        t, r = events(df, k, thr, hold, direction)
                        T.append(t)
                        R.append(r)
                        B.append(baseline(df, hold, -direction))
                    t, r = np.concatenate(T), np.concatenate(R)
                    if len(r) < 20:
                        continue
                    net = r - cost
                    tt, nd = day_t(t, net)
                    h1, h2 = net[t < mid], net[t >= mid]
                    rows.append((f"{name} ≥{100 * thr:.0f}% за {kname}", hname, len(net), nd, (r > 0).mean(), net.mean(),
                                 float(np.mean(B)) - cost, tt, h1.mean() if len(h1) else np.nan, h2.mean() if len(h2) else np.nan))
    for ev, hn, n, nd, w, m, b, tt, h1, h2 in rows:
        good = m > 0 and h1 > 0 and h2 > 0 and tt > 3
        print(f"  {ev:<24}{hn:>7}{n:>8}{nd:>6}{100 * w:>8.1f}%{100 * m:>10.3f}%{100 * b:>7.3f}%{tt:>9.1f}"
              f"{100 * h1:>9.3f}%{100 * h2:>9.3f}%" + ("  ← устойчивый плюс" if good else ""))
    best = [r for r in rows if r[5] > 0 and r[8] > 0 and r[9] > 0 and r[7] > 3]
    print(f"\nПроверено вариантов: {len(rows)}. Устойчивых (плюс после издержек в обеих половинах и t > 3): {len(best)}.")
    print("Как читать: «в среднем» > 0 — после комиссии в плюсе. При десятках вариантов несколько штук с t≈2 появятся случайно;"
          "\nверить можно только строкам «устойчивый плюс», и даже их надо проверить вперёд (на новых данных), прежде чем торговать.")


if __name__ == "__main__":
    main()
