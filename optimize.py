"""Перебор таймфреймов и параметров на истории.

    python optimize.py --days 180 --timeframes 1h,4h --symbols BTC/USDT,ETH/USDT,TON/USDT [--synthetic]

Данные каждой пары делятся на обучающую (70%) и проверочную (30%) части. «Проходит» набор,
который прибылен и на обучающей, и на проверочной части, при достаточном числе сделок на проверочной.
Один «удачный» результат на одной паре ничего не доказывает (при переборе он возможен случайно):
доверять стоит только наборам, проходящим на нескольких парах.
"""
import argparse
from collections import defaultdict
from dataclasses import replace

import ccxt

import config
from backtest import run, run_futures
from config import Params
from data import fetch_history, synthetic
from exchanges import make_exchange

EMA_PAIRS = [(5, 20), (10, 30), (20, 50)]
STOP_TAKE = [(0.01, 0.02), (0.02, 0.04), (0.03, 0.06)]


def stats(df, p, runner=run):
    final, trades, curve = runner(df, p)
    start = config.START_BALANCE
    dd = ((curve.cummax() - curve) / curve.cummax()).max() if len(curve) else 0
    wins = sum(t > 0 for t in trades)
    return {"ret": (final / start - 1) * 100, "trades": len(trades),
            "win": wins / len(trades) * 100 if trades else 0, "dd": dd * 100}


def passes(r, min_trades):
    return r["train"]["ret"] > 0 and r["test"]["ret"] > 0 and r["test"]["trades"] >= min_trades


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--timeframes", default="15m,1h,4h", help="через запятую: 5m,15m,1h,4h,1d")
    ap.add_argument("--symbols", default=config.SYMBOL, help="через запятую: BTC/USDT,ETH/USDT")
    ap.add_argument("--min-trades", type=int, default=10, help="мин. сделок на проверочной части")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--market", default=config.MARKET, choices=["spot", "futures"])
    a = ap.parse_args()
    ex = None if a.synthetic else make_exchange(config.EXCHANGE, market=a.market)
    runner = run_futures if a.market == "futures" else run
    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]

    results = {}          # symbol -> список наборов
    for si, sym in enumerate(symbols):
        print(f"\n=== {sym} ===")
        rows = []
        for tf in a.timeframes.split(","):
            try:
                df = synthetic(3000, seed=si + 1) if a.synthetic else fetch_history(ex, sym, tf, a.days)
            except ccxt.ExchangeError as e:
                print(f"{tf}: пропущено ({str(e)[:90]})")
                continue
            if len(df) < 200:
                print(f"{tf}: слишком мало свечей ({len(df)}), пропущено")
                continue
            cut = int(len(df) * 0.7)
            train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
            bh = (test.iloc[-1].close / test.iloc[0].close - 1) * 100
            print(f"{tf}: свечей {len(df)}, B&H на проверочной части {bh:+.1f}%")
            for (f, s), (sl, tp) in [(e, st) for e in EMA_PAIRS for st in STOP_TAKE]:
                p = replace(Params(), ema_fast=f, ema_slow=s, stop_loss_pct=sl, take_profit_pct=tp)
                rows.append({"key": (tf, f, s, sl, tp), "train": stats(train, p, runner),
                             "test": stats(test, p, runner)})
        if not rows:
            continue
        results[sym] = rows
        rows.sort(key=lambda r: r["train"]["ret"], reverse=True)
        print(f"Топ-5 по обучающей части (проверочная — честная оценка):")
        print(f"{'TF':<4}{'EMA':<8}{'стоп/тейк':<11}{'train%':>8}{'сделок':>7}{'| test%':>9}{'сделок':>7}{'винрейт':>9}{'просадка':>9}")
        for r in rows[:5]:
            tf, f, s, sl, tp = r["key"]; tr, te = r["train"], r["test"]
            print(f"{tf:<4}{f'{f}/{s}':<8}{f'{sl*100:.0f}/{tp*100:.0f}%':<11}{tr['ret']:>8.1f}{tr['trades']:>7}"
                  f"{'|':>2}{te['ret']:>7.1f}{te['trades']:>7}{te['win']:>8.0f}%{te['dd']:>8.1f}%")

    if not results:
        raise SystemExit("\nНет данных ни по одной паре.")

    print("\n" + "=" * 70)
    print(f"СВОДКА (набор «проходит»: прибыль на обеих частях и ≥{a.min_trades} сделок на проверочной)")
    print(f"{'пара':<14}{'наборов':>8}{'проходят':>10}{'лучший test%':>14}")
    passing = defaultdict(list)
    for sym, rows in results.items():
        ok = [r for r in rows if passes(r, a.min_trades)]
        for r in ok:
            passing[r["key"]].append(sym)
        best = max(r["test"]["ret"] for r in rows)
        print(f"{sym:<14}{len(rows):>8}{len(ok):>10}{best:>13.1f}%")

    robust = {k: v for k, v in passing.items() if len(v) >= 2}
    print("\nУстойчивые наборы (проходят минимум на 2 парах):")
    if robust:
        for (tf, f, s, sl, tp), syms in sorted(robust.items(), key=lambda kv: -len(kv[1])):
            print(f"  {tf}  EMA {f}/{s}  стоп/тейк {sl*100:.0f}/{tp*100:.0f}%  -> {', '.join(syms)}")
    else:
        print("  нет. Это значит, что видимые «плюсы» (если есть) скорее всего случайны.")
    total = sum(len(v) for v in results.values())
    print(f"\nВсего проверок: {total}. При случайной стратегии часть из них «пройдёт» просто по везению — "
          "поэтому смотрите на устойчивые наборы, а не на единичные.")
