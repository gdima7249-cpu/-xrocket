"""Перебор таймфреймов и параметров на истории: python optimize.py --days 180 --timeframes 1h,4h [--synthetic]

Данные делятся на обучающую (70%) и проверочную (30%) части: параметры
выбираются по обучающей, а честная оценка — по проверочной. Если результат
на проверочной части сильно хуже, это переобучение и доверять ему нельзя.
"""
import argparse
from dataclasses import replace

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--timeframes", default="15m,1h,4h", help="через запятую: 5m,15m,1h,4h,1d")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--market", default=config.MARKET, choices=["spot", "futures"])
    a = ap.parse_args()
    ex = None if a.synthetic else make_exchange(config.EXCHANGE, market=a.market)

    runner = run_futures if a.market == "futures" else run
    rows = []
    for tf in a.timeframes.split(","):
        df = synthetic(3000) if a.synthetic else fetch_history(ex, config.SYMBOL, tf, a.days)
        cut = int(len(df) * 0.7)
        train, test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
        bh = (test.iloc[-1].close / test.iloc[0].close - 1) * 100
        print(f"{tf}: свечей {len(df)}, B&H на проверочной части {bh:+.1f}%")
        for (f, s), (sl, tp) in [(e, st) for e in EMA_PAIRS for st in STOP_TAKE]:
            p = replace(Params(), ema_fast=f, ema_slow=s, stop_loss_pct=sl, take_profit_pct=tp)
            rows.append((tf, f, s, sl, tp, stats(train, p, runner), stats(test, p, runner), bh))

    rows.sort(key=lambda r: r[5]["ret"], reverse=True)
    print("\nТоп-10 по обучающей части (проверочная — честная оценка):")
    print(f"{'TF':<4}{'EMA':<8}{'стоп/тейк':<11}{'train%':>8}{'сделок':>7}{'| test%':>9}{'сделок':>7}{'винрейт':>9}{'просадка':>9}")
    for tf, f, s, sl, tp, tr, te, _ in rows[:10]:
        print(f"{tf:<4}{f'{f}/{s}':<8}{f'{sl*100:.0f}/{tp*100:.0f}%':<11}{tr['ret']:>8.1f}{tr['trades']:>7}"
              f"{'|':>2}{te['ret']:>7.1f}{te['trades']:>7}{te['win']:>8.0f}%{te['dd']:>8.1f}%")
    good = [r for r in rows if r[5]["ret"] > 0 and r[6]["ret"] > 0 and r[6]["trades"] >= 5]
    print(f"\nНаборов, прибыльных и на обучающей, и на проверочной части: {len(good)} из {len(rows)}")
