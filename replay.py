"""Быстрый прогон тех же стратегий, что работают в радаре и multi_paper, по годам истории — за секунды.

Берёт часовые/4-часовые свечи Binance (публично, ключи не нужны), считает сигналы без заглядывания вперёд,
ведёт сделки по тем же правилам (тейк / стоп / таймаут / сигнал выхода, вход по открытию следующей свечи),
учитывает издержки круга и печатает ту же таблицу, что `radar.py --stats`, плюс проверки:
по месяцам, первая/вторая половина периода, контроль на перемешанных свечах, чувствительность к издержкам.

  python replay.py                         # 20 пар, 365 дней, издержки круга 0.4% (измерены на крупных биржах)
  python replay.py --days 730 --fee 0.001 --slip 0.001
  python replay.py --synthetic             # проверка кода без интернета (случайные данные)

Ограничения: история ≠ будущее; внутри свечи, если задеты и стоп, и тейк, считается стоп (консервативно);
реальный стакан в истории недоступен — издержки задаются числом.
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config
from optimize import shuffled
from radar import STRATS, stats_text

API = "https://data-api.binance.vision/api/v3/klines"
CACHE = "research_cache"
TF_HOURS = {"1h": 1, "4h": 4}
PAIRS = "BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,DOT,LTC,TRX,NEAR,ATOM,UNI,XLM,APT,ARB,OP,TON".split(",")


def download(symbol: str, tf: str, days: int) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{symbol}_{tf}_{days}d.csv"
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
        return pd.read_csv(path)
    end = int(time.time() * 1000)
    since = end - days * 86400 * 1000
    step = TF_HOURS[tf] * 3600 * 1000
    rows = []
    while since < end:
        r = requests.get(API, params={"symbol": symbol, "interval": tf, "startTime": since, "limit": 1000}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + step
        time.sleep(0.05)
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df = df.astype(float).drop_duplicates("time").iloc[:-1]            # без незакрытой свечи
    df.to_csv(path, index=False)
    return df


def synthetic(days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    open_ = np.r_[close[0], close[:-1]]
    t0 = int(time.time() * 1000) - n * 3600 * 1000
    return pd.DataFrame({"time": t0 + np.arange(n) * 3600 * 1000.0, "open": open_,
                         "high": np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n))),
                         "low": np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n))),
                         "close": close, "volume": rng.uniform(1, 10, n)})


def resample4h(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["time"] // (4 * 3600 * 1000))
    return pd.DataFrame({"time": g["time"].first(), "open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum()}
                        ).reset_index(drop=True)


def simulate(df: pd.DataFrame, en: np.ndarray, ex: np.ndarray, tp: float, sl: float, hold_bars: int):
    """Сделки без издержек: [(индекс входа, индекс выхода, цена входа, цена выхода, причина)]."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n, i, out = len(df), 0, []
    while i < n - 1:
        if not en[i]:
            i += 1
            continue
        e_idx, entry = i + 1, o[i + 1]
        j, done = e_idx, False
        while j < n:
            sl_p, tp_p = entry * (1 - sl), entry * (1 + tp)
            if l[j] <= sl_p:
                out.append((e_idx, j, entry, sl_p if j == e_idx else min(o[j], sl_p), "stop-loss")); done = True
            elif h[j] >= tp_p:
                out.append((e_idx, j, entry, max(o[j], tp_p) if j > e_idx else tp_p, "take-profit")); done = True
            elif j - e_idx + 1 >= hold_bars:
                out.append((e_idx, j, entry, c[j], "timeout")); done = True
            elif ex[j]:
                out.append((e_idx, j, entry, o[j + 1] if j + 1 < n else c[j], "signal")); done = True
            if done:
                i = j
                break
            j += 1
        if not done:
            break                                               # позиция не закрылась к концу истории — не считаем
    return out


def run_all(frames: dict, fee: float, slip: float):
    """frames: {sym: {'1h': df, '4h': df}} -> список закрытых сделок в формате радара + сырые цены."""
    closed = []
    for sym, tfs in frames.items():
        for st, cfg in STRATS.items():
            df = tfs.get(cfg["tf"])
            if df is None or len(df) < 120:
                continue
            en, ex = cfg["fn"](df.reset_index(drop=True))
            hold = max(1, int(cfg["hold"] / TF_HOURS[cfg["tf"]]))
            t = df["time"].to_numpy() / 1000
            for e_idx, x_idx, pe, px, why in simulate(df, en.to_numpy(bool), ex.to_numpy(bool), cfg["tp"], cfg["sl"], hold):
                closed.append({"s": st, "sym": sym, "reason": why, "t0": float(t[e_idx]), "t": float(t[x_idx]),
                               "pe": pe, "px": px, "ret": net(pe, px, fee, slip)})
    return closed


def net(pe, px, fee, slip):
    return px * (1 - slip) * (1 - fee) / (pe * (1 + slip) * (1 + fee)) - 1


def summarize(rows):
    r = [x["ret"] * 100 for x in rows]
    return (len(r), 100 * sum(v > 0 for v in r) / len(r), sum(r) / len(r)) if r else (0, 0.0, 0.0)


def month(c):
    return datetime.fromtimestamp(c["t0"]).strftime("%Y-%m")


def selftest():
    """Проверка расчёта на вручную составленных свечах: правильный ответ известен заранее."""
    def candles(rows):
        return pd.DataFrame([{"time": i * 3600_000.0, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}
                             for i, (o, h, l, c) in enumerate(rows)])

    def one(rows, entry_at, exit_at=(), tp=0.05, sl=0.03, hold=100):
        df = candles(rows)
        en = np.zeros(len(df), bool); en[entry_at] = True
        ex = np.zeros(len(df), bool)
        for k in exit_at:
            ex[k] = True
        return simulate(df, en, ex, tp, sl, hold)

    flat = (100, 100, 100, 100)
    checks = []
    # 1. тейк: вход по открытию свечи 2 (=100), свеча 3 задевает 105 -> выход ровно 105
    t = one([flat, flat, flat, (100, 106, 99, 104), flat], entry_at=1)
    checks.append(("тейк-профит срабатывает по цене тейка", t and t[0][4] == "take-profit" and abs(t[0][3] - 105) < 1e-9))
    # 2. стоп: свеча 3 падает до 96 -> выход 97
    t = one([flat, flat, flat, (100, 101, 96, 98), flat], entry_at=1)
    checks.append(("стоп-лосс срабатывает по цене стопа", t and t[0][4] == "stop-loss" and abs(t[0][3] - 97) < 1e-9))
    # 3. и стоп, и тейк в одной свече -> считается стоп
    t = one([flat, flat, flat, (100, 107, 95, 100), flat], entry_at=1)
    checks.append(("стоп и тейк в одной свече -> стоп", t and t[0][4] == "stop-loss"))
    # 4. гэп вниз за стоп: открытие 90, стоп 97 -> выход по 90 (хуже стопа)
    t = one([flat, flat, flat, (90, 91, 89, 90), flat], entry_at=1)
    checks.append(("гэп вниз: выход по цене открытия, не по стопу", t and abs(t[0][3] - 90) < 1e-9))
    # 5. таймаут
    t = one([flat] * 8, entry_at=1, hold=3)
    checks.append(("таймаут закрывает позицию", t and t[0][4] == "timeout" and t[0][1] - t[0][0] == 2))
    # 6. сигнал выхода: сигнал на свече 4 -> выход по открытию свечи 5
    rows = [flat, flat, flat, flat, flat, (101, 101, 101, 101), flat]
    t = one(rows, entry_at=1, exit_at=[4])
    checks.append(("сигнал выхода: цена открытия следующей свечи", t and t[0][4] == "signal" and abs(t[0][3] - 101) < 1e-9))
    # 7. вход по открытию СЛЕДУЮЩЕЙ свечи после сигнала (без заглядывания вперёд)
    rows = [(100, 100, 100, 100), (110, 111, 109, 110), flat, flat]
    t = one(rows, entry_at=0, tp=0.5, sl=0.5, hold=1)
    checks.append(("вход по открытию следующей свечи", t and abs(t[0][2] - 110) < 1e-9))
    # 8. позиция, не закрывшаяся к концу истории, не считается
    t = one([flat] * 5, entry_at=1, tp=0.5, sl=0.5)
    checks.append(("незакрытая к концу позиция не учитывается", t == []))
    # 9. издержки: цена не изменилась, fee 0.1% и slip 0.1% с каждой стороны -> убыток ≈ 0.4%
    r = net(100, 100, 0.001, 0.001)
    checks.append(("издержки круга 0.4%", abs(r + 0.004) < 0.0001))
    # 10. +5% на тейке при тех же издержках ≈ +4.6%
    r = net(100, 105, 0.001, 0.001)
    checks.append(("прибыль 5% минус издержки ≈ 4.6%", abs(r - 0.0460) < 0.0005))

    bad = 0
    for name, ok in checks:
        print(("  ok   " if ok else "  ОШИБКА  ") + name)
        bad += not ok
    print(f"\nИтог: {len(checks) - bad}/{len(checks)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="проверить правильность расчёта на ручных примерах")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--fee", type=float, default=0.001, help="комиссия за сторону (0.001 = 0.1%%)")
    ap.add_argument("--slip", type=float, default=0.001, help="проскальзывание за сторону")
    ap.add_argument("--control-runs", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()

    frames = {}
    for i, s in enumerate(a.pairs.split(",")):
        try:
            if a.synthetic:
                h1 = synthetic(a.days, i)
                frames[s] = {"1h": h1, "4h": resample4h(h1)}
            else:
                frames[s] = {tf: download(s + "USDT", tf, a.days) for tf in ("1h", "4h")}
            print(f"{s}: {len(frames[s]['1h'])} часовых свечей", flush=True)
        except Exception as e:
            print(f"{s}: пропуск ({type(e).__name__}: {e})")
    if not frames:
        raise SystemExit("Нет данных (проверьте доступ к data-api.binance.vision)")

    bh = [100 * (f["1h"]["close"].iloc[-1] / f["1h"]["close"].iloc[0] - 1) for f in frames.values()]
    print(f"\nПериод: {a.days} дн., пар {len(frames)}. Просто купить и держать (среднее по парам): {np.mean(bh):+.0f}%."
          f" Издержки круга: {200 * (a.fee + a.slip):.2f}%.\n")

    closed = run_all(frames, a.fee, a.slip)
    print(stats_text({"open": {}, "closed": closed}))

    print("\nПо месяцам входа (все стратегии отдельно: сделок / win% / ср.% на сделку):")
    months = sorted({month(c) for c in closed})
    print(f"  {'месяц':<9}" + "".join(f"{st:>20}" for st in STRATS))
    for m in months:
        cells = []
        for st in STRATS:
            n, w, avg = summarize([c for c in closed if c["s"] == st and month(c) == m])
            cells.append(f"{n:>5} {w:>4.0f}% {avg:>+6.2f}" if n else f"{'-':>20}")
        print(f"  {m:<9}" + "".join(f"{x:>20}" for x in cells))

    print("\nКонтроль на перемешанных свечах (тот же рост и волатильность, без «тайминга»): сделок / win% / ср.% на сделку")
    for st in STRATS:
        ctrl = []
        for seed in range(1, a.control_runs + 1):
            shuf = {sym: {tf: shuffled(d.reset_index(drop=True), seed) for tf, d in tfs.items()}
                    for sym, tfs in frames.items()}
            ctrl += [c for c in run_all(shuf, a.fee, a.slip) if c["s"] == st]
        n, w, avg = summarize(ctrl)
        n_real, w_real, avg_real = summarize([c for c in closed if c["s"] == st])
        print(f"  {st:<15} реальные: {n_real:>5} {w_real:>4.0f}% {avg_real:>+6.2f}   контроль (в среднем на прогон): "
              f"{n // a.control_runs:>5} {w:>4.0f}% {avg:>+6.2f}")

    print("\nЧувствительность к издержкам круга (ср.% на сделку):")
    costs = [0.002, 0.004, 0.008, 0.011]
    print(f"  {'стратегия':<15}" + "".join(f"{100 * c:>8.1f}%" for c in costs))
    for st in STRATS:
        rows = [c for c in closed if c["s"] == st]
        vals = [100 * np.mean([net(c["pe"], c["px"], cst / 4, cst / 4) for c in rows]) if rows else float("nan") for cst in costs]
        print(f"  {st:<15}" + "".join(f"{v:>+9.2f}" for v in vals))

    print("\nВЕРДИКТ по стратегиям (нужно всё сразу: 200+ сделок, 20+ дней входа, ср.% > 0, обе половины периода > 0,"
          " плюс в 60%+ месяцев, лучше контроля):")
    mid = np.median([c["t0"] for c in closed]) if closed else 0
    for st in STRATS:
        rows = [c for c in closed if c["s"] == st]
        if not rows:
            print(f"  {st:<15} нет сделок"); continue
        n, w, avg = summarize(rows)
        days = len({datetime.fromtimestamp(c["t0"]).date() for c in rows})
        h1, h2 = summarize([c for c in rows if c["t0"] <= mid])[2], summarize([c for c in rows if c["t0"] > mid])[2]
        mm = [summarize([c for c in rows if month(c) == m])[2] for m in months if any(month(c) == m for c in rows)]
        pos_m = np.mean([v > 0 for v in mm]) if mm else 0
        ok = n >= 200 and days >= 20 and avg > 0 and h1 > 0 and h2 > 0 and pos_m >= 0.6
        print(f"  {st:<15} сделок {n}, дней {days}, ср.% {avg:+.2f}, половины {h1:+.2f}/{h2:+.2f}, "
              f"плюсовых месяцев {100 * pos_m:.0f}%  -> {'КАНДИДАТ' if ok else 'не проходит'}")
    print("\nКАНДИДАТ ≠ деньги: сначала проверка на другом периоде/парах и живой журнал. Контроль должен быть заметно хуже реального.")


if __name__ == "__main__":
    main()
