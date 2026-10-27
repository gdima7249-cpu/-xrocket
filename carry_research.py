"""Оффлайн-проверка «фандинг-арбитража» (cash-and-carry): спот-лонг + фьючерс-шорт на одну сумму, цена монеты не важна,
доход — платежи фандинга (раз в 8 часов; когда ставка положительна, лонги платят шортам).

Берём реальную историю ставок фандинга Binance (публично, ключи не нужны) и считаем:
  1) сколько приносил фандинг по каждой монете сам по себе (в годовых процентах от суммы позиции);
  2) портфель «лучшие K монет по недавнему фандингу»: раз в неделю пересобираем, платим комиссии за вход/выход, держим только монеты с
     положительным недавним фандингом, остальное — в деньгах (без дохода);
  3) результат на весь капитал: нужно держать и спот, и маржу под шорт, поэтому доходность на капитал = фандинг / коэффициент капитала
     (по умолчанию 2.0 = плечо 1x на фьючерсе; 1.33 при плече 3x, но тогда выше риск ликвидации шорта при резком росте цены).

НЕ учтено: разница цен спота и фьючерса (базис) при входе/выходе, риск ликвидации шорта при резком росте, риск биржи, налоги, ограничения
на размер. Это верхняя оценка спокойной доходности, не гарантия.

  python carry_research.py                          # 42 монеты, 7 лет
  python carry_research.py --top 5 --lookback 21 --capital-factor 2
  python carry_research.py --synthetic              # проверка кода без интернета
"""
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd

from replay import PAIRS
from replay_ls import fetch_funding

ALL = ("BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,DOT,LTC,TRX,NEAR,ATOM,UNI,XLM,APT,ARB,OP,TON,FIL,AAVE,ICP,ETC,BCH,SAND,MANA,GRT,CRV,SUSHI,"
       "ALGO,EGLD,THETA,XTZ,EOS,NEO,IOTA,VET,HBAR,ZEC,DASH,KAVA").split(",")
STEP = 8 * 3600


def load(symbols, days, synthetic=False):
    """DataFrame: индекс — время (сек, кратно 8 часам), столбцы — монеты, значения — ставка фандинга за период."""
    series = {}
    for s in symbols:
        try:
            if synthetic:
                rng = np.random.default_rng(abs(hash(s)) % 10_000)
                t = np.arange(int(time.time()) // STEP * STEP - days * 86400, int(time.time()), STEP, dtype=float)
                regime = np.sin(np.arange(len(t)) / 400.0)                # то «жадный» рынок, то «боязливый»
                r = 0.0001 + 0.00006 * regime + rng.normal(0, 0.00004, len(t))
            else:
                t, r = fetch_funding(s + "USDT", days)
            idx = (np.round(np.asarray(t) / STEP) * STEP).astype(np.int64)
            series[s] = pd.Series(np.asarray(r), index=idx).groupby(level=0).last()
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {str(e)[:70]})")
    return pd.DataFrame(series).sort_index()


def simulate(F, top, lookback_days, rebalance_days, cost_roundtrip, min_avg):
    """Возвращает массив доходностей на каждый период (в долях от суммы позиции) для портфеля лучших монет."""
    n = len(F)
    look, reb = lookback_days * 3, rebalance_days * 3                       # периодов в сутках: 3
    ret = np.zeros(n)
    held, invested = [], False
    for i in range(n):
        if i >= look and (i - look) % reb == 0:
            avg = F.iloc[i - look:i].mean()
            cand = avg[avg > min_avg].sort_values(ascending=False).head(top).index.tolist()
            changed = set(cand) ^ set(held)
            if changed:
                # платим комиссии за каждую вошедшую/вышедшую монету (вход или выход = половина круга), долю от всего портфеля
                w = 1 / max(top, 1)
                ret[i] -= len(changed) * w * (cost_roundtrip / 2)
            held = cand
        if held:
            row = F.iloc[i][held].fillna(0.0)
            ret[i] += row.sum() / max(top, 1)                               # равные доли на top монет (часть в деньгах, если монет меньше)
    return ret


def summarize(ret, idx, capital_factor):
    cap = ret / capital_factor                                              # на весь капитал (спот + маржа под шорт)
    eq = np.cumprod(1 + cap)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1).min()
    years = (idx[-1] - idx[0]) / (365.25 * 86400)
    cagr = eq[-1] ** (1 / years) - 1
    by_year = {}
    t = pd.to_datetime(idx, unit="s")
    for y in sorted(set(t.year)):
        m = t.year == y
        by_year[y] = np.prod(1 + cap[m]) - 1
    return eq[-1], cagr, dd, by_year


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=2555)
    ap.add_argument("--pairs", default=",".join(ALL))
    ap.add_argument("--top", type=int, default=5, help="сколько лучших монет держать одновременно")
    ap.add_argument("--lookback", type=int, default=21, help="за сколько дней смотрим средний фандинг при выборе")
    ap.add_argument("--rebalance", type=int, default=7, help="как часто пересобираем портфель, дней")
    ap.add_argument("--cost", type=float, default=0.003, help="издержки полного круга по паре спот+фьючерс (комиссии на 4 ордера, доля от позиции)")
    ap.add_argument("--min-avg", type=float, default=0.00005, help="мин. средний фандинг за период (0.00005 = 0.005%% за 8 ч), иначе монету не берём")
    ap.add_argument("--capital-factor", type=float, default=2.0, help="капитал / сумма позиции: 2.0 при плече 1x, 1.33 при 3x")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    F = load(a.pairs.split(","), a.days, a.synthetic)
    if F.empty:
        raise SystemExit("Нет данных по фандингу (проверьте доступ к fapi.binance.com)")
    idx = F.index.to_numpy()
    print(f"Фандинг загружен для {F.shape[1]} монет, {len(F)} периодов (по 8 ч), {datetime.fromtimestamp(idx[0]):%Y-%m-%d} … {datetime.fromtimestamp(idx[-1]):%Y-%m-%d}.\n")

    print("1) Средний фандинг по монетам (если просто держать спот+шорт на всём периоде), годовых % от суммы позиции:")
    ann = (F.mean() * 3 * 365 * 100).sort_values(ascending=False)
    print("   лучшие:  " + "  ".join(f"{k} {v:+.1f}%" for k, v in ann.head(6).items()))
    print("   худшие:  " + "  ".join(f"{k} {v:+.1f}%" for k, v in ann.tail(4).items()))
    print(f"   по всем монетам в среднем: {ann.mean():+.1f}% годовых от суммы позиции (до комиссий и без выбора)\n")

    print(f"2) Портфель: лучшие {a.top} монет по среднему фандингу за {a.lookback} дн., пересборка раз в {a.rebalance} дн., "
          f"издержки круга {100 * a.cost:.2f}% на монету, капитал/позиция = {a.capital_factor}:")
    ret = simulate(F, a.top, a.lookback, a.rebalance, a.cost, a.min_avg)
    eq, cagr, dd, by_year = summarize(ret, idx, a.capital_factor)
    print(f"   итог {eq:.2f}x, доходность {100 * cagr:+.1f}% в год на весь капитал, макс. просадка (по закрытым платежам) {100 * dd:.1f}%")
    print("   по годам: " + "  ".join(f"{y}: {100 * v:+.1f}%" for y, v in by_year.items()))
    share_in = float((ret != 0).mean())
    print(f"   доля времени с позицией ≈ {100 * share_in:.0f}%\n")

    print("3) Чувствительность к издержкам и капиталу (доходность в год на капитал):")
    print(f"   {'издержки круга':<18}" + "".join(f"{f'капитал x{cf}':>26}" for cf in (2.0, 1.5, 1.33)))
    for c in (0.001, 0.003, 0.006):
        row = []
        for cf in (2.0, 1.5, 1.33):
            e, g, d, _ = summarize(simulate(F, a.top, a.lookback, a.rebalance, c, a.min_avg), idx, cf)
            row.append(f"{100 * g:+.1f}% (просадка {100 * d:.1f}%)")
        print(f"   {f'{100 * c:.1f}%':<18}" + "".join(f"{x:>26}" for x in row))
    print("\nОграничения: не учтены базис (разница цен спота и фьючерса), ликвидация шорта при резком росте цены (держите запас маржи),"
          " риск биржи, налоги. Это верхняя оценка спокойной доходности.")


if __name__ == "__main__":
    main()
