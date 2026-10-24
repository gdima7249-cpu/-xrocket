"""Сравнение стратегий на истории по нескольким парам (только спот, только покупка).

    python optimize.py --days 180 --timeframes 1h,4h --symbols BTC/USDT,ETH/USDT,GRAM/USDT [--synthetic]

Стратегии: ema_rsi (базовая), trend_pullback (откат в тренде), donchian (пробой канала), bollinger (возврат к среднему).
Данные пары: обучающая (70%) и проверочная (30%) части — для таблицы, а для вердикта история делится на три
равных последовательных периода. Набор «проходит», если прибылен в КАЖДОМ из трёх периодов (так «плюс» нельзя
получить, просто сидя в растущем рынке) и сделок суммарно достаточно. Доверять стоит только наборам, проходящим
минимум на 2 парах (но крипто-пары ходят вместе, так что и это не гарантия).
"""
import argparse
from collections import defaultdict
from dataclasses import replace

import ccxt
import numpy as np

import config
from backtest import run_signals
from config import Params
from data import fetch_history, synthetic
from exchanges import make_exchange
from strategies import GRID, STOP_TAKE


def stats(df, p, en, ex_):
    final, trades, curve = run_signals(df, p, en, ex_)
    dd = ((curve.cummax() - curve) / curve.cummax()).max() if len(curve) else 0
    wins = sum(t > 0 for t in trades)
    return {"ret": (final / config.START_BALANCE - 1) * 100, "trades": len(trades),
            "win": wins / len(trades) * 100 if trades else 0, "dd": dd * 100}


def passes(r, min_trades):
    return all(s["ret"] > 0 for s in r["segs"]) and sum(s["trades"] for s in r["segs"]) >= min_trades


def label(kw):
    return ",".join(f"{k}={v}" for k, v in kw.items())


def evaluate(df, tf):
    """Все стратегии и параметры на одной истории -> список результатов."""
    cut = int(len(df) * 0.7)
    b3 = [0, len(df) // 3, 2 * len(df) // 3, len(df)]
    rows = []
    for name, fn, kw in GRID:
        en, ex_ = fn(df, **kw)                # сигналы на полной истории (только прошлые данные)
        for sl, tp in STOP_TAKE:
            p = replace(Params(), stop_loss_pct=sl, take_profit_pct=tp)
            tr = stats(df.iloc[:cut], p, en.iloc[:cut], ex_.iloc[:cut])
            te = stats(df.iloc[cut:], p, en.iloc[cut:], ex_.iloc[cut:])
            segs = [stats(df.iloc[b3[j]:b3[j + 1]], p, en.iloc[b3[j]:b3[j + 1]], ex_.iloc[b3[j]:b3[j + 1]])
                    for j in range(3)]
            rows.append({"key": (name, tf, label(kw), f"{sl*100:.0f}/{tp*100:.0f}%"),
                         "train": tr, "test": te, "segs": segs})
    return rows


def shuffled(df, seed):
    """Контроль: те же свечи, но в случайном порядке (тот же рост и волатильность, без «тайминга»)."""
    rng = np.random.default_rng(seed)
    c = df.close.values
    r = np.log(c[1:] / c[:-1])
    idx = rng.permutation(len(r))
    close = c[0] * np.exp(np.r_[0.0, np.cumsum(r[idx])])
    hi_rel = (df.high.values / c)[1:][idx]
    lo_rel = (df.low.values / c)[1:][idx]
    out = df.copy()
    out["close"] = close
    out["open"] = np.r_[close[0], close[:-1]]
    out["high"] = np.r_[close[0], close[1:] * hi_rel]
    out["low"] = np.r_[close[0], close[1:] * lo_rel]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--timeframes", default="1h,4h", help="через запятую: 15m,1h,4h,1d")
    ap.add_argument("--symbols", default=config.SYMBOL, help="через запятую: BTC/USDT,ETH/USDT")
    ap.add_argument("--min-trades", type=int, default=20, help="мин. сделок суммарно по трём периодам")
    ap.add_argument("--null-runs", type=int, default=3, help="сколько контрольных прогонов на перемешанных данных")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--market", default="spot", choices=["spot", "futures"])
    a = ap.parse_args()
    if a.market == "futures":
        raise SystemExit("Сравнение стратегий пока только для спота (только покупка).")
    ex = None if a.synthetic else make_exchange(config.EXCHANGE)
    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]

    results, null_rows = {}, []
    for si, sym in enumerate(symbols):
        print(f"\n=== {sym} ===")
        rows = []
        for tf in a.timeframes.split(","):
            try:
                df = synthetic(3000, seed=si + 1, drift=0.0) if a.synthetic else fetch_history(ex, sym, tf, a.days)
            except ccxt.ExchangeError as e:
                print(f"{tf}: пропущено ({str(e)[:90]})")
                continue
            if len(df) < 400:
                print(f"{tf}: слишком мало свечей ({len(df)}), пропущено")
                continue
            cut = int(len(df) * 0.7)
            bh = (df.iloc[-1].close / df.iloc[cut].close - 1) * 100
            print(f"{tf}: свечей {len(df)}, B&H на проверочной части {bh:+.1f}%")
            rows += evaluate(df, tf)
            for k in range(a.null_runs):
                null_rows += evaluate(shuffled(df, 1000 * (si + 1) + k), tf)
        if not rows:
            continue
        results[sym] = rows
        rows.sort(key=lambda r: r["train"]["ret"], reverse=True)
        print("Топ-8 по обучающей части (проверочная — честная оценка):")
        print(f"{'стратегия':<15}{'TF':<4}{'параметры':<24}{'ст/тейк':<8}{'train%':>8}{'сд.':>4}{'| test%':>9}{'сд.':>4}{'win%':>6}{'просад':>7}  3 периода, %")
        for r in rows[:8]:
            n, tf, lb, st = r["key"]; tr, te = r["train"], r["test"]
            print(f"{n:<15}{tf:<4}{lb:<24}{st:<8}{tr['ret']:>8.1f}{tr['trades']:>4}"
                  f"{'|':>2}{te['ret']:>7.1f}{te['trades']:>4}{te['win']:>6.0f}{te['dd']:>7.1f}  "
                  + " / ".join(f"{s['ret']:+.1f}" for s in r["segs"]))

    if not results:
        raise SystemExit("\nНет данных ни по одной паре.")

    print("\n" + "=" * 78)
    print(f"СВОДКА (набор «проходит»: прибыль в каждом из трёх периодов и ≥{a.min_trades} сделок суммарно)")
    names = sorted({k[0] for rows in results.values() for r in rows for k in [r["key"]]})
    print(f"{'стратегия':<16}" + "".join(f"{s:>14}" for s in results) + f"{'всего проходит':>16}")
    passing = defaultdict(list)
    for n in names:
        line, total = f"{n:<16}", 0
        for sym, rows in results.items():
            mine = [r for r in rows if r["key"][0] == n]
            ok = [r for r in mine if passes(r, a.min_trades)]
            for r in ok:
                passing[r["key"]].append(sym)
            total += len(ok)
            line += f"{f'{len(ok)} из {len(mine)}':>14}"
        print(line + f"{total:>16}")

    real_rows = [r for rows in results.values() for r in rows]
    real_rate = sum(passes(r, a.min_trades) for r in real_rows) / len(real_rows) * 100
    null_rate = sum(passes(r, a.min_trades) for r in null_rows) / max(len(null_rows), 1) * 100
    print(f"\nКОНТРОЛЬ: доля проходящих наборов на реальных данных {real_rate:.1f}%, "
          f"на перемешанных (случайных, но с тем же ростом и волатильностью) {null_rate:.1f}%.")
    if real_rate <= null_rate * 1.5 + 2:
        print("  -> на реальных данных проходит не заметно больше, чем на случайных: стратегии неотличимы от случайности.")
    else:
        print("  -> на реальных данных проходит заметно больше, чем на случайных: есть повод проверить глубже "
              "(другие периоды и пары).")

    robust = {k: v for k, v in passing.items() if len(v) >= 2}
    print("\nУстойчивые наборы (проходят минимум на 2 парах):")
    if robust:
        for (n, tf, lb, st), syms in sorted(robust.items(), key=lambda kv: -len(kv[1])):
            print(f"  {n} {tf} [{lb}] стоп/тейк {st} -> {', '.join(syms)}")
    else:
        print("  нет. Видимые «плюсы» (если есть) скорее всего случайны.")
    total = sum(len(v) for v in results.values())
    print(f"\nВсего проверок: {total}. При таком количестве часть наборов «пройдёт» по везению — "
          "ориентируйтесь на устойчивые, а не на единичные.")
