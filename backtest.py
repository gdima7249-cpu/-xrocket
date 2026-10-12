"""Бэктест: python backtest.py [--days 365] [--synthetic] [--csv file.csv]"""
import argparse

import pandas as pd

import config
from config import Params
from exchanges import make_exchange
from data import fetch_history, synthetic
from strategy import add_indicators, position_size, signal


def run(df: pd.DataFrame, p: Params, balance: float = config.START_BALANCE):
    df = add_indicators(df, p)
    cash, qty, entry = balance, 0.0, 0.0
    trades, equity_curve = [], []

    for i in range(p.ema_slow + 2, len(df)):
        row = df.iloc[i]

        # Выход по стопу/тейку внутри свечи (стоп приоритетнее — консервативно)
        if qty > 0:
            stop, take = entry * (1 - p.stop_loss_pct), entry * (1 + p.take_profit_pct)
            exit_price = None
            if row.low <= stop:
                exit_price = stop
            elif row.high >= take:
                exit_price = take
            if exit_price:
                exit_price *= 1 - p.slippage
                cash += qty * exit_price * (1 - p.fee)
                trades.append(cash_pnl(entry, exit_price, qty, p))
                qty = 0.0

        # Сигнал по закрытой свече -> сделка по цене закрытия
        sig = signal(df.iloc[: i + 1], p)
        if sig == "buy" and qty == 0:
            equity = cash
            price = row.close * (1 + p.slippage)
            q = position_size(equity, price, p)
            q = min(q, cash / (price * (1 + p.fee)))
            if q > 0:
                cash -= q * price * (1 + p.fee)
                qty, entry = q, price
        elif sig == "sell" and qty > 0:
            price = row.close * (1 - p.slippage)
            cash += qty * price * (1 - p.fee)
            trades.append(cash_pnl(entry, price, qty, p))
            qty = 0.0

        equity_curve.append(cash + qty * row.close)

    final = cash + qty * df.iloc[-1].close
    return final, trades, pd.Series(equity_curve)


def cash_pnl(entry, exit_, qty, p):
    return qty * (exit_ * (1 - p.fee) - entry * (1 + p.fee))


def report(start, final, trades, curve, df):
    wins = [t for t in trades if t > 0]
    dd = ((curve.cummax() - curve) / curve.cummax()).max() if len(curve) else 0
    bh = df.iloc[-1].close / df.iloc[0].close - 1
    print(f"Свечей:           {len(df)}")
    print(f"Сделок:           {len(trades)}")
    print(f"Винрейт:          {len(wins) / len(trades) * 100:.1f}%" if trades else "Винрейт:          -")
    print(f"Итог:             {start:.2f} -> {final:.2f} ({(final / start - 1) * 100:+.2f}%)")
    print(f"Макс. просадка:   {dd * 100:.2f}%")
    print(f"Buy & Hold:       {bh * 100:+.2f}%  (для сравнения)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()

    if a.synthetic:
        df = synthetic()
    elif a.csv:
        df = pd.read_csv(a.csv, parse_dates=["time"])
    else:
        ex = make_exchange(config.EXCHANGE)
        df = fetch_history(ex, config.SYMBOL, config.TIMEFRAME, a.days)
        df.to_csv("history.csv", index=False)

    p = Params()
    final, trades, curve = run(df, p)
    report(config.START_BALANCE, final, trades, curve, df)
