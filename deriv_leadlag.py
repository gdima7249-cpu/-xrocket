"""«Лидер → догоняющий» на рынках Deriv: если одна пара резко дёрнулась за 5 минут, а связанная с ней ещё стоит на месте,
догонит ли она? (Та же идея, что «BTC → альткоины», но на валютах, металлах и крипте Deriv.)

Для каждой пары рынков (лидер L, догоняющий F):
  сигнал — L сдвинулся сильнее k «обычных разбросов» за одну 5-минутную свечу, а F за ту же свечу почти не сдвинулся (меньше 1);
  ставка — на F в сторону, куда его обычно тянет за L (по знаку их корреляции за прошлые ~7 дней; нужна |корреляция| ≥ 0.3).
Честная проверка: правила ищутся на ПЕРВОЙ половине истории, проверяются на ВТОРОЙ. Можно входить с задержкой (--delay 1):
если «догонялка» — это рассинхрон котировок на секунды, при входе через 5 минут она исчезнет.

  python deriv_leadlag.py                      # валюты, металлы, BTC и ETH (свечи из кэша research_cache, если не старше 7 дней)
  python deriv_leadlag.py --delay 1            # вход через 5 минут после сигнала
Ничего не покупает.
"""
import argparse
import math
import time

import numpy as np

import deriv_forex_test as F
from deriv_scan import FOREX, wilson_lo

GRAN = 300
KS = (2.5, 3.0, 4.0)
HORIZONS = (3, 6)                                                    # 15 и 30 минут


def load(symbols, candles, max_age):
    data = {}
    for j, s in enumerate(symbols):
        try:
            ts, c = F.fetch_candles(s, candles, max_age=max_age)
        except (SystemExit, Exception) as e:  # noqa: BLE001
            print(f"{s}: пропускаю — {type(e).__name__}: {str(e)[:120]}")
            continue
        if F.LAST_DOWNLOADED and j < len(symbols) - 1:
            time.sleep(30)
        if len(c) >= 5000:
            data[s] = (np.asarray(ts, dtype=np.int64), np.asarray(c, dtype=float))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="", help="свой список через запятую")
    ap.add_argument("--payout", type=float, default=0.78)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--min-bets", type=int, default=40)
    ap.add_argument("--candles", type=int, default=60000)
    ap.add_argument("--max-age-days", type=float, default=7)
    a = ap.parse_args()
    import pandas as pd
    syms = a.symbols.split(",") if a.symbols else FOREX + ["cryBTCUSD", "cryETHUSD"]
    data = load(syms, a.candles, a.max_age_days * 86400)
    be = 1 / (1 + a.payout)
    d = a.delay
    # общая сетка времени: только моменты, когда торгуются ОБЕ пары, проверяется для каждой пары отдельно
    allt = np.unique(np.concatenate([ts for ts, _ in data.values()]))
    pos = {t: i for i, t in enumerate(allt)}
    n = len(allt)
    C, R, Z = {}, {}, {}
    for s, (ts, c) in data.items():
        cc = np.full(n, np.nan)
        cc[[pos[t] for t in ts]] = c
        C[s] = cc
        r = np.r_[np.nan, np.diff(np.log(cc))]
        r[np.r_[False, np.diff(allt) != GRAN]] = np.nan                # через перерыв во времени — не изменение за 5 минут
        R[s] = r
        Z[s] = r / pd.Series(r).rolling(2000, min_periods=500).std().to_numpy()
    hf = ((allt + GRAN) % 86400) / 3600
    half = n // 2
    print(f"\nРынков: {len(data)}. Пар «лидер → догоняющий»: {len(data) * (len(data) - 1)}. "
          f"Порог безубыточности при выплате {100 * a.payout:.0f}%: {100 * be:.1f}%."
          + (f" Вход через {5 * d} мин после сигнала." if d else ""))
    t0 = time.time()
    rows = []
    names = list(data)
    for L in names:
        for Fs in names:
            if L == Fs:
                continue
            corr = pd.Series(R[L]).rolling(2000, min_periods=500).corr(pd.Series(R[Fs])).shift(1).to_numpy()
            zl, zf, cf = Z[L], Z[Fs], C[Fs]
            for h in HORIZONS:
                end = np.full(n, np.nan)
                st = np.full(n, np.nan)
                end[:n - h - d] = cf[h + d:]
                st[:n - h - d] = cf[d:n - h]
                ok_t = np.zeros(n, bool)
                ok_t[:n - h - d] = (allt[h + d:] - allt[:n - h - d]) == (h + d) * GRAN   # без перерывов до выхода
                pause = (hf < 22) & (hf + (h + d) * 5 / 60 > 20 + 50 / 60)
                base = ok_t & ~pause & (np.abs(zf) < 1) & (np.abs(corr) >= 0.3) & ~np.isnan(end) & ~np.isnan(st)
                for k in KS:
                    m = base & (np.abs(zl) >= k)
                    idx, last = [], -10 ** 9
                    for i in np.flatnonzero(m):
                        if i > last + h + d:
                            idx.append(i)
                            last = i
                    if len(idx) < 2 * a.min_bets:
                        continue
                    idx = np.array(idx)
                    want = np.sign(zl[idx]) * np.sign(corr[idx])         # куда F обычно идёт за L
                    win = np.sign(end[idx] - st[idx]) == want
                    s1, s2 = idx < half, idx >= half
                    w1, n1, w2, n2 = int(win[s1].sum()), int(s1.sum()), int(win[s2].sum()), int(s2.sum())
                    if n1 < a.min_bets or w1 / n1 < be + 0.04:
                        continue
                    good = n2 >= a.min_bets and w2 / n2 > be and wilson_lo(w2, n2) > be
                    rows.append((L, Fs, h, k, w1, n1, w2, n2, good))
    print(f"Готово за {time.time() - t0:.0f} с. На первой половине выше порога (+4%): {len(rows)}; "
          f"подтвердились на второй: {sum(r[8] for r in rows)}.")
    print("(На стольких парах часть случайно проходит первую половину — верим только подтверждённым.)\n")

    def show(r):
        L, Fs, h, k, w1, n1, w2, n2, good = r
        return (f"  {'✅' if good else '  '} {L.replace('frx', '').replace('cry', ''):>7} → {Fs.replace('frx', '').replace('cry', ''):<7}"
                f" рывок ≥{k:g}, {h * 5} мин │ 1-я половина {100 * w1 / n1:4.1f}%/{n1:<4} │ 2-я половина "
                f"{100 * w2 / max(n2, 1):4.1f}%/{n2:<4} (не ниже {100 * wilson_lo(w2, n2):4.1f}%)")

    good = sorted([r for r in rows if r[8]], key=lambda r: -wilson_lo(r[6], r[7]))
    if good:
        print("ПОДТВЕРДИЛИСЬ на второй половине:")
        for r in good[:30]:
            print(show(r))
    print("\nСамые «красивые» на первой половине (для сравнения):")
    for r in sorted(rows, key=lambda r: -r[4] / r[5])[:15]:
        print(show(r))


if __name__ == "__main__":
    main()
