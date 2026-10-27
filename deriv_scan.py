"""Сканер правил «резкое движение → откат» по реальным рынкам Deriv с честной проверкой на данных, которых он не видел.

Перебирает сотни правил (падение/взлёт, сила, тренд за 12 ч, часы суток, срок ставки) и защищается от «миражей»:
  1) пары делятся на две группы: ПОИСК и ПРОВЕРКА (через одну по алфавиту);
  2) история каждой пары делится пополам по времени: первая половина и вторая;
  3) правила ищутся ТОЛЬКО на «поиск, первая половина»;
  4) найденное проверяется на трёх кусках, которых поиск не видел: поиск/2-я половина, проверка/1-я, проверка/2-я.
Остаётся то, что выше порога безубыточности на всех трёх кусках, а вместе — с запасом больше погрешности.

Как бот: сигнал на закрытии 5-минутной свечи, вход по её цене; пока ставка по паре открыта, новые сигналы по ней не берутся;
«цена не изменилась» — проигрыш; ставки, которые задели бы перерыв Deriv (20:50–22:00 UTC), и окна с перерывом в торговле
(ночь, выходные) не считаются. Разброс — по прошлым 2000 свечам, без заглядывания вперёд.

  python deriv_scan.py                        # все валюты и металлы (свечи из кэша research_cache, если не старше 7 дней)
  python deriv_scan.py --with-indices         # плюс индексы
  python deriv_scan.py --payout 0.78          # выплата для порога безубыточности (по умолчанию 0.78 → 56.2%)
Ничего не покупает. Итог печатается и сохраняется в scan_results.csv.
"""
import argparse
import csv
import math
import time

import numpy as np

import deriv_forex_test as F

GRAN = 300
FOREX = ["frxAUDCAD", "frxAUDCHF", "frxAUDJPY", "frxAUDNZD", "frxAUDUSD", "frxEURAUD", "frxEURCAD", "frxEURCHF", "frxEURGBP",
         "frxEURJPY", "frxEURNZD", "frxEURUSD", "frxGBPAUD", "frxGBPCAD", "frxGBPCHF", "frxGBPJPY", "frxGBPNZD", "frxGBPUSD",
         "frxNZDJPY", "frxNZDUSD", "frxUSDCAD", "frxUSDCHF", "frxUSDJPY", "frxUSDMXN", "frxUSDPLN",
         "frxXAGUSD", "frxXAUUSD", "frxXPDUSD", "frxXPTUSD"]
INDICES = ["OTC_AEX", "OTC_AS51", "OTC_DJI", "OTC_FCHI", "OTC_FTSE", "OTC_GDAXI", "OTC_HSI", "OTC_N225", "OTC_NDX", "OTC_SPC",
           "OTC_SSMI", "OTC_SX5E"]
KS = (2.0, 2.5, 3.0, 4.0)
HOURS = [(a, a + 3) for a in range(0, 24, 3)] + [(0, 24)]           # блоки по 3 часа (UTC) и «любое время»
HORIZONS = (3, 6, 12)                                                # 15 / 30 / 60 минут
TRENDS = ("вверх", "вниз", "любой")


def prepare(ts, c):
    """Все величины, которые бот видит в момент сигнала (свеча i закрылась), и итог ставки через h свечей."""
    ts, c = np.asarray(ts, dtype=np.int64), np.asarray(c, dtype=float)
    lc = np.log(c)
    ok_step = np.r_[False, np.diff(ts) == GRAN]                     # свеча i идёт сразу за i-1 (без перерыва)
    gaps = np.cumsum(~ok_step)                                       # номер «куска без перерывов»
    r = np.r_[np.nan, np.diff(lc)]
    r[~ok_step] = np.nan
    import pandas as pd
    vol = pd.Series(r).rolling(2000, min_periods=500).std().to_numpy()
    n = len(c)
    i = np.arange(n)
    z = np.full(n, np.nan)
    v3 = (i >= 3)
    same3 = np.zeros(n, bool)
    same3[3:] = gaps[3:] == gaps[:-3]
    z[v3] = (lc[3:] - lc[:-3]) / (vol[3:] * math.sqrt(3))
    z[~same3] = np.nan
    tr = np.full(n, np.nan)
    same144 = np.zeros(n, bool)
    same144[144:] = gaps[144:] == gaps[:-144]
    tr[144:] = lc[141:-3] - lc[:-144]
    tr[~same144] = np.nan
    close_h = ((ts + GRAN) % 86400) / 3600                           # час закрытия сигнальной свечи (как в боте)
    return {"ts": ts, "c": c, "gaps": gaps, "z": z, "tr": tr, "hf": close_h, "n": n}


def outcomes(p, h):
    """win[i] — ставка ВВЕРХ от закрытия свечи i выиграла через h свечей; valid[i] — окно без перерывов и без перерыва Deriv."""
    n, c, gaps, hf = p["n"], p["c"], p["gaps"], p["hf"]
    valid = np.zeros(n, bool)
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    valid[:n - h] = gaps[h:] == gaps[:n - h]
    up[:n - h] = c[h:] > c[:n - h]
    dn[:n - h] = c[h:] < c[:n - h]
    pause = (hf < 22) & (hf + h * 5 / 60 > 20 + 50 / 60)             # ставка задела бы перерыв Deriv
    valid &= ~pause
    return valid, up, dn


def bets(mask, h):
    """Как бот: пока ставка по паре открыта (h свечей), новые сигналы по ней не берутся."""
    out, last = [], -10 ** 9
    for i in np.flatnonzero(mask):
        if i > last + h:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def rule_name(kind, k, trend, hrs, h):
    lo, hi = hrs
    when = "любое время" if (lo, hi) == (0, 24) else f"{(lo + 3) % 24:02d}–{(hi + 3) % 24 or 24:02d} мск"
    what = f"падение ≥{k:g} → ВВЕРХ" if kind == "dip" else f"взлёт ≥{k:g} → ВНИЗ"
    return f"{what}, тренд12 {trend}, {when}, {h * 5} мин"


def evaluate(prep, kind, k, trend, hrs, h):
    """Возвращает {(группа, половина): [выиграно, ставок]} для одного правила."""
    res = {}
    for sym, (grp, p, oc) in prep.items():
        valid, up, dn = oc[h]
        z, tr, hf = p["z"], p["tr"], p["hf"]
        m = (z <= -k) if kind == "dip" else (z >= k)
        if trend == "вверх":
            m &= tr > 0
        elif trend == "вниз":
            m &= tr < 0
        lo, hi = hrs
        if (lo, hi) != (0, 24):
            m &= (hf >= lo) & (hf < hi)
        m &= valid
        idx = bets(m, h)
        if not len(idx):
            continue
        win = (up if kind == "dip" else dn)[idx]
        half = p["n"] // 2
        for part, sel in (("1", idx < half), ("2", idx >= half)):
            t = res.setdefault((grp, part), [0, 0])
            t[0] += int(win[sel].sum())
            t[1] += int(sel.sum())
    return res


def wilson_lo(w, n, zc=1.96):
    if not n:
        return 0.0
    p = w / n
    d = 1 + zc * zc / n
    return (p + zc * zc / (2 * n) - zc * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n))) / d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="", help="свой список через запятую (по умолчанию все валюты и металлы)")
    ap.add_argument("--with-indices", action="store_true")
    ap.add_argument("--payout", type=float, default=0.78, help="выплата для порога безубыточности (0.78 → 56.2%%)")
    ap.add_argument("--min-bets", type=int, default=60, help="минимум ставок на куске поиска")
    ap.add_argument("--candles", type=int, default=60000)
    ap.add_argument("--max-age-days", type=float, default=7, help="брать свечи из кэша, если они не старше стольких дней")
    a = ap.parse_args()
    syms = a.symbols.split(",") if a.symbols else FOREX + (INDICES if a.with_indices else [])
    be = 1 / (1 + a.payout)
    prep = {}
    order = sorted(syms)
    for j, s in enumerate(order):
        try:
            ts, c = F.fetch_candles(s, a.candles, max_age=a.max_age_days * 86400)
        except (SystemExit, Exception) as e:  # noqa: BLE001
            print(f"{s}: пропускаю — {type(e).__name__}: {str(e)[:120]}")
            continue
        if F.LAST_DOWNLOADED and j < len(order) - 1:
            time.sleep(30)                                           # не упираться в лимит Deriv
        if len(c) < 5000:
            print(f"{s}: мало свечей, пропуск")
            continue
        p = prepare(ts, c)
        prep[s] = ("поиск" if j % 2 == 0 else "проверка", p, {h: outcomes(p, h) for h in HORIZONS})
    groups = {g: [s for s, v in prep.items() if v[0] == g] for g in ("поиск", "проверка")}
    print(f"\nПары для ПОИСКА ({len(groups['поиск'])}): {', '.join(x.replace('frx', '') for x in groups['поиск'])}")
    print(f"Пары для ПРОВЕРКИ ({len(groups['проверка'])}): {', '.join(x.replace('frx', '') for x in groups['проверка'])}")
    print(f"Порог безубыточности при выплате {100 * a.payout:.0f}%: {100 * be:.1f}%\n")

    rules = [(kind, k, tr, hrs, h) for kind in ("dip", "pump") for k in KS for tr in TRENDS for hrs in HOURS for h in HORIZONS]
    print(f"Перебираю {len(rules)} правил...")
    t0 = time.time()
    rows = []
    for rl in rules:
        res = evaluate(prep, *rl)
        w0, n0 = res.get(("поиск", "1"), [0, 0])
        if n0 < a.min_bets or w0 / n0 < be + 0.04:                   # на куске поиска нужно хотя бы +4% к порогу
            continue
        checks = [res.get(key, [0, 0]) for key in (("поиск", "2"), ("проверка", "1"), ("проверка", "2"))]
        wc, nc = sum(x[0] for x in checks), sum(x[1] for x in checks)
        all_above = all(n and w / n > be for w, n in checks)
        good = all_above and nc >= 100 and wilson_lo(wc, nc) > be
        rows.append((rl, w0, n0, checks, wc, nc, good))
    print(f"Готово за {time.time() - t0:.0f} с.")

    passed = [r for r in rows if r[6]]
    print(f"\nНа куске ПОИСКА выше порога (+4%) оказалось {len(rows)} правил из {len(rules)}. "
          f"Из них прошли проверку на невиденных данных: {len(passed)}.")
    print("(Даже на случайных ценах часть правил проходит поиск — поэтому верим только прошедшим проверку.)\n")

    def show(r):
        rl, w0, n0, checks, wc, nc, good = r
        parts = "  ".join(f"{100 * w / n:4.1f}%/{n}" if n else "  —  " for w, n in checks)
        return (f"  {'✅' if good else '  '} {rule_name(*rl):<58} поиск {100 * w0 / n0:4.1f}%/{n0:<4} │ проверка: {parts} │ "
                f"вместе {100 * wc / max(nc, 1):4.1f}% (не ниже {100 * wilson_lo(wc, nc):4.1f}%) на {nc}")

    if passed:
        print("ПРОШЛИ ПРОВЕРКУ (лучшие по нижней границе на невиденных данных):")
        for r in sorted(passed, key=lambda r: -wilson_lo(r[4], r[5]))[:40]:
            print(show(r))
    print("\nСамые «красивые» на поиске (для сравнения — видно, как проценты падают на проверке):")
    for r in sorted(rows, key=lambda r: -r[1] / r[2])[:15]:
        print(show(r))
    with open("scan_results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["правило", "поиск_угадано", "поиск_ставок", "проверка_угадано", "проверка_ставок", "прошло"])
        for r in rows:
            wr.writerow([rule_name(*r[0]), r[1], r[2], r[4], r[5], int(r[6])])
    print("\nВсё сохранено в scan_results.csv. Колонки проверки: поиск/2-я половина, проверка/1-я, проверка/2-я (угадано/ставок).")
    print("Важно: часы указаны по Москве (UTC+3). Найденное я сверю с тем, что уже есть в боте, прежде чем что-то включать.")


if __name__ == "__main__":
    main()
