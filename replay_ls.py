"""Быстрый прогон стратегий в ОБЕ стороны (long и short) по годам истории — как replay.py, но с шортами.

За год рынок мог падать, а покупать на падении невыгодно; шорт (ставка на снижение) в такие периоды зарабатывает.
Проверяем три классических трендовых подхода (4-часовые свечи Binance, 20 пар):
  donchian_ls — пробой канала вверх (лонг) / вниз (шорт), выход по обратному каналу
  ema_ls      — EMA20 выше EMA50: лонг, ниже: шорт (всегда в рынке, разворот по пересечению)
  tsmom_ls    — 30-дневная доходность > 0: лонг, < 0: шорт (моментум)

Модель: вход по открытию следующей свечи, стоп/тейк внутри свечи (при касании обоих — стоп), таймаут, комиссия+проскальзывание
с каждой стороны. НЕ учтены: плата за займ/фандинг фьючерса (в падающем рынке шорт обычно получает фандинг, в растущем платит),
ликвидации при плече (здесь плечо 1).

  python replay_ls.py                  # 20 пар, 1095 дней, издержки круга 0.4%
  python replay_ls.py --days 730
  python replay_ls.py --selftest       # проверка расчёта на ручных свечах
  python replay_ls.py --synthetic [--plant]   # без интернета; --plant вшивает тренд, чтобы проверить, что детектор его видит
"""
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from optimize import shuffled
from replay import PAIRS, TF_HOURS, download, resample4h, synthetic

BARS_PER_DAY = 6                                   # 4-часовые свечи


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _valid(df, warmup):
    return pd.Series(np.arange(len(df)) >= warmup, index=df.index)


def donchian_ls(df, n_in=40, n_out=20):
    hi_in, lo_in = df.high.rolling(n_in).max().shift(1), df.low.rolling(n_in).min().shift(1)
    hi_out, lo_out = df.high.rolling(n_out).max().shift(1), df.low.rolling(n_out).min().shift(1)
    v = _valid(df, max(n_in, n_out) + 1)
    return ((df.close > hi_in) & v, df.close < lo_out, (df.close < lo_in) & v, df.close > hi_out)


def ema_ls(df, fast=20, slow=50):
    f, s = _ema(df.close, fast), _ema(df.close, slow)
    v = _valid(df, slow + 2)
    up = (f.shift(1) <= s.shift(1)) & (f > s) & v
    dn = (f.shift(1) >= s.shift(1)) & (f < s) & v
    return up, dn, dn, up                            # разворот: сигнал шорта закрывает лонг и наоборот


def tsmom_ls(df, days=30):
    n = days * BARS_PER_DAY
    r = df.close / df.close.shift(n) - 1
    v = _valid(df, n + 1)
    up = (r > 0) & ~(r.shift(1) > 0) & v
    dn = (r < 0) & ~(r.shift(1) < 0) & v
    return up, dn, dn, up


STRATS = {                                         # (функция, тейк, стоп, таймаут в свечах)
    "donchian_ls": (donchian_ls, 0.20, 0.06, 6 * 30),
    "ema_ls":      (ema_ls,      0.20, 0.06, 6 * 30),
    "tsmom_ls":    (tsmom_ls,    0.20, 0.06, 6 * 30),
}


def simulate(df, le, lx, se, sx, tp, sl, hold):
    """Сделки без издержек: [(сторона +1/-1, индекс входа, индекс выхода, цена входа, цена выхода, причина)]."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n, i, out = len(df), 0, []
    while i < n - 1:
        side = 1 if le[i] else (-1 if se[i] else 0)
        if not side:
            i += 1
            continue
        e_idx, entry = i + 1, o[i + 1]
        sl_p = entry * (1 - sl) if side > 0 else entry * (1 + sl)
        tp_p = entry * (1 + tp) if side > 0 else entry * (1 - tp)
        j, done = e_idx, False
        while j < n:
            hit_sl = l[j] <= sl_p if side > 0 else h[j] >= sl_p
            hit_tp = h[j] >= tp_p if side > 0 else l[j] <= tp_p
            xs = lx[j] if side > 0 else sx[j]
            if hit_sl:
                gap = (min(o[j], sl_p) if side > 0 else max(o[j], sl_p)) if j > e_idx else sl_p
                out.append((side, e_idx, j, entry, gap, "stop-loss")); done = True
            elif hit_tp:
                px = (max(o[j], tp_p) if side > 0 else min(o[j], tp_p)) if j > e_idx else tp_p
                out.append((side, e_idx, j, entry, px, "take-profit")); done = True
            elif j - e_idx + 1 >= hold:
                out.append((side, e_idx, j, entry, c[j], "timeout")); done = True
            elif xs:
                out.append((side, e_idx, j, entry, o[j + 1] if j + 1 < n else c[j], "signal")); done = True
            if done:
                i = j
                break
            j += 1
        if not done:
            break
    return out


def net(side, pe, px, fee, slip):
    if side > 0:
        return px * (1 - slip) * (1 - fee) / (pe * (1 + slip) * (1 + fee)) - 1
    return pe * (1 - slip) * (1 - fee) / (px * (1 + slip) * (1 + fee)) - 1      # шорт: продали дорого, откупили дёшево


def run_all(frames, fee, slip):
    closed = []
    for sym, df in frames.items():
        if len(df) < 400:
            continue
        df = df.reset_index(drop=True)
        t = df["time"].to_numpy() / 1000
        for st, (fn, tp, sl, hold) in STRATS.items():
            le, lx, se, sx = (x.to_numpy(bool) for x in fn(df))
            for side, e_idx, x_idx, pe, px, why in simulate(df, le, lx, se, sx, tp, sl, hold):
                closed.append({"s": st, "sym": sym, "side": side, "reason": why, "t0": float(t[e_idx]),
                               "pe": pe, "px": px, "ret": net(side, pe, px, fee, slip)})
    return closed


def summ(rows):
    r = [x["ret"] * 100 for x in rows]
    return (len(r), 100 * sum(v > 0 for v in r) / len(r), sum(r) / len(r)) if r else (0, 0.0, 0.0)


def month(c):
    return datetime.fromtimestamp(c["t0"]).strftime("%Y-%m")


def selftest():
    def candles(rows):
        return pd.DataFrame([{"time": i * 3600_000.0, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}
                             for i, (o, h, l, c) in enumerate(rows)])

    def one(rows, side, at=1, tp=0.05, sl=0.03, hold=100, exit_at=()):
        df = candles(rows)
        z = np.zeros(len(df), bool)
        le, se, lx, sx = z.copy(), z.copy(), z.copy(), z.copy()
        (le if side > 0 else se)[at] = True
        for k in exit_at:
            (lx if side > 0 else sx)[k] = True
        return simulate(df, le, lx, se, sx, tp, sl, hold)

    flat = (100, 100, 100, 100)
    checks = []
    t = one([flat, flat, flat, (100, 101, 94, 95), flat], -1)
    checks.append(("шорт: цена упала на 5% -> тейк по цене тейка (95)", t and t[0][5] == "take-profit" and abs(t[0][4] - 95) < 1e-9))
    t = one([flat, flat, flat, (100, 104, 99, 103), flat], -1)
    checks.append(("шорт: цена выросла на 3% -> стоп по цене стопа (103)", t and t[0][5] == "stop-loss" and abs(t[0][4] - 103) < 1e-9))
    t = one([flat, flat, flat, (108, 109, 107, 108), flat], -1)
    checks.append(("шорт: гэп вверх за стоп -> выход по открытию (108), не по стопу", t and abs(t[0][4] - 108) < 1e-9))
    t = one([flat, flat, flat, (100, 107, 95, 100), flat], -1)
    checks.append(("шорт: стоп и тейк в одной свече -> стоп", t and t[0][5] == "stop-loss"))
    t = one([flat, flat, flat, (100, 106, 99, 104), flat], 1)
    checks.append(("лонг: тейк по цене тейка (105)", t and t[0][5] == "take-profit" and abs(t[0][4] - 105) < 1e-9))
    checks.append(("прибыль шорта: вход 100, выход 95, издержки 0.4% -> ≈ +4.84%", abs(net(-1, 100, 95, 0.001, 0.001) - 0.0484) < 0.0003))
    checks.append(("убыток шорта: вход 100, выход 103 -> ≈ -3.30%", abs(net(-1, 100, 103, 0.001, 0.001) + 0.0330) < 0.0003))
    checks.append(("лонг без движения: -0.4%", abs(net(1, 100, 100, 0.001, 0.001) + 0.004) < 0.0001))
    checks.append(("шорт без движения: -0.4%", abs(net(-1, 100, 100, 0.001, 0.001) + 0.004) < 0.0001))
    bad = 0
    for name, ok in checks:
        print(("  ok   " if ok else "  ОШИБКА  ") + name)
        bad += not ok
    print(f"\nИтог: {len(checks) - bad}/{len(checks)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def make_synth(days, seed, plant):
    df = synthetic(days, seed)
    if plant:                                      # вшитый тренд: автокорреляция доходностей
        rng = np.random.default_rng(seed + 100)
        r = np.zeros(len(df))
        for k in range(1, len(df)):
            r[k] = 0.97 * r[k - 1] + rng.normal(0, 0.0015)
        close = 100 * np.exp(np.cumsum(r + rng.normal(0, 0.006, len(df))))
        df["open"] = np.r_[close[0], close[:-1]]
        df["close"] = close
        df["high"] = np.maximum(df["open"], close) * 1.002
        df["low"] = np.minimum(df["open"], close) * 0.998
    return resample4h(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--control-runs", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="для --synthetic: другой набор случайных данных")
    a = ap.parse_args()
    if a.selftest:
        selftest()

    frames = {}
    for i, s in enumerate(a.pairs.split(",")):
        try:
            frames[s] = make_synth(a.days, i + 1000 * a.seed, a.plant) if a.synthetic else download(s + "USDT", "4h", a.days)
            gaps = int((np.diff(frames[s]["time"].to_numpy()) != 4 * 3600 * 1000).sum())
            print(f"{s}: {len(frames[s])} 4-часовых свечей" + (f"  (!) пропусков в данных: {gaps}" if gaps else ""), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {e})")
    if not frames:
        raise SystemExit("Нет данных (проверьте доступ к data-api.binance.vision)")

    bh = [100 * (f["close"].iloc[-1] / f["close"].iloc[0] - 1) for f in frames.values()]
    print(f"\nПериод: {a.days} дн., пар {len(frames)}. Купить и держать: {np.mean(bh):+.0f}%, "
          f"продать и держать (шорт): {-np.mean(bh):+.0f}% (грубо). Издержки круга: {200 * (a.fee + a.slip):.2f}%.\n")
    closed = run_all(frames, a.fee, a.slip)

    print(f"{'стратегия':<14}{'сделок':>8}{'win%':>6}{'ср.%':>8}{'итого%':>9}   лонги: сделок / ср.%   шорты: сделок / ср.%")
    for st in STRATS:
        rows = [c for c in closed if c["s"] == st]
        n, w, avg = summ(rows)
        nl, _, al = summ([c for c in rows if c["side"] > 0])
        ns, _, as_ = summ([c for c in rows if c["side"] < 0])
        print(f"{st:<14}{n:>8}{w:>6.0f}{avg:>8.2f}{sum(c['ret'] for c in rows) * 100:>9.1f}   {nl:>14} / {al:+.2f}   {ns:>14} / {as_:+.2f}")

    print("\nУстойчивость к выбросам (если результат держится на единичных гигантских сделках — это не стратегия):")
    print(f"  {'стратегия':<14}{'среднее':>9}{'медиана':>9}{'без 1% лучших и худших':>24}{'без 10 лучших':>15}{'доля прибыли от 10 лучших':>27}")
    for st in STRATS:
        r = np.sort(np.array([c["ret"] * 100 for c in closed if c["s"] == st]))
        if len(r) < 50:
            continue
        k = max(1, len(r) // 100)
        total_pos = r[r > 0].sum()
        print(f"  {st:<14}{r.mean():>+9.2f}{np.median(r):>+9.2f}{r[k:-k].mean():>+24.2f}{r[:-10].mean():>+15.2f}"
              f"{100 * r[-10:].sum() / total_pos if total_pos else 0:>26.0f}%")
    top = sorted(closed, key=lambda c: -c["ret"])[:8]
    print("\n8 самых прибыльных сделок (проверьте, нет ли тут сбоя данных):")
    for c in top:
        print(f"  {c['s']:<12}{c['sym']:<6}{'лонг' if c['side'] > 0 else 'шорт':<5}"
              f"{datetime.fromtimestamp(c['t0']):%Y-%m-%d %H:%M}  вход {c['pe']:.6g} -> выход {c['px']:.6g}  {c['ret'] * 100:+.1f}%  ({c['reason']})")

    months = sorted({month(c) for c in closed})
    print("\nПо месяцам входа (сделок / ср.% на сделку):")
    print(f"  {'месяц':<9}" + "".join(f"{st:>20}" for st in STRATS))
    for m in months:
        cells = []
        for st in STRATS:
            n, _, avg = summ([c for c in closed if c["s"] == st and month(c) == m])
            cells.append(f"{n:>6} {avg:>+8.2f}" if n else "-")
        print(f"  {m:<9}" + "".join(f"{x:>20}" for x in cells))

    print("\nКонтроль на перемешанных свечах (без «тайминга», тот же рост/волатильность): сделок / win% / ср.%")
    ctrl_avg = {}
    for st in STRATS:
        ctrl = []
        for seed in range(1, a.control_runs + 1):
            shuf = {sym: shuffled(d.reset_index(drop=True), seed) for sym, d in frames.items()}
            ctrl += [c for c in run_all(shuf, a.fee, a.slip) if c["s"] == st]
        n, w, avg = summ(ctrl)
        ctrl_avg[st] = avg
        nr, wr, ar = summ([c for c in closed if c["s"] == st])
        print(f"  {st:<14} реальные: {nr:>5} {wr:>4.0f}% {ar:>+6.2f}   контроль: {n // a.control_runs:>5} {w:>4.0f}% {avg:>+6.2f}")

    print("\nЧувствительность к издержкам круга (ср.% на сделку):")
    costs = [0.002, 0.004, 0.008, 0.011]
    print(f"  {'стратегия':<14}" + "".join(f"{100 * c:>8.1f}%" for c in costs))
    for st in STRATS:
        rows = [c for c in closed if c["s"] == st]
        print(f"  {st:<14}" + "".join(
            f"{100 * np.mean([net(c['side'], c['pe'], c['px'], k / 4, k / 4) for c in rows]):>+9.2f}" if rows else f"{'-':>9}"
            for k in costs))

    print("\nВЕРДИКТ (всё сразу: 200+ сделок, 20+ дней входа, ср.% > 0, обе половины > 0, плюс в 60%+ месяцев,"
          " t-статистика (кластеры по дням) > 3 и лучше контроля минимум на 0.5 п.п.):")
    mid = np.median([c["t0"] for c in closed]) if closed else 0
    for st in STRATS:
        rows = [c for c in closed if c["s"] == st]
        if not rows:
            print(f"  {st:<14} нет сделок"); continue
        n, w, avg = summ(rows)
        days = len({datetime.fromtimestamp(c["t0"]).date() for c in rows})
        h1, h2 = summ([c for c in rows if c["t0"] <= mid])[2], summ([c for c in rows if c["t0"] > mid])[2]
        mm = [summ([c for c in rows if month(c) == m])[2] for m in months if any(month(c) == m for c in rows)]
        pos = np.mean([v > 0 for v in mm]) if mm else 0
        # t-статистика среднего по сделкам с кластеризацией по дням: сделки одного дня зависимы (рынок двигается общий),
        # поэтому ошибку считаем по суммам отклонений внутри дня
        by_day = {}
        for c in rows:
            by_day.setdefault(datetime.fromtimestamp(c["t0"]).date(), []).append(c["ret"] * 100)
        dev_sums = np.array([np.sum(np.array(v) - avg) for v in by_day.values()])
        se = np.sqrt((dev_sums ** 2).sum()) / n
        t_stat = avg / se if se > 0 else 0.0
        better = avg - ctrl_avg.get(st, 0.0)
        ok = (n >= 200 and days >= 20 and avg > 0 and h1 > 0 and h2 > 0 and pos >= 0.6
              and t_stat > 3 and better >= 0.5)
        print(f"  {st:<14} сделок {n}, дней {days}, ср.% {avg:+.2f}, половины {h1:+.2f}/{h2:+.2f}, "
              f"плюсовых месяцев {100 * pos:.0f}%, t={t_stat:+.1f}, лучше контроля на {better:+.2f} -> "
              f"{'КАНДИДАТ' if ok else 'не проходит'}")
    print("\nКАНДИДАТ ≠ деньги: сначала проверка на другом периоде/парах, учёт фандинга и живой журнал.")


if __name__ == "__main__":
    main()
