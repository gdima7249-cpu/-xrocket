"""Бэктест: python backtest.py [--days 365] [--synthetic] [--csv file.csv]"""
import argparse

import pandas as pd

import config
from config import Params
from exchanges import make_exchange
from data import fetch_history, synthetic
from strategy import add_indicators, cross, position_size, position_size_futures, signal


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


def run_futures(df: pd.DataFrame, p: Params, balance: float = config.START_BALANCE,
                leverage: float = config.LEVERAGE):
    """Бэктест long+short с плечом. Стоп/тейк внутри свечи (стоп приоритетнее)."""
    df = add_indicators(df, p)
    cash, side, qty, entry, open_fee = balance, None, 0.0, 0.0, 0.0
    trades, curve = [], []

    def close(price):
        nonlocal cash, side, qty, entry
        sign = 1 if side == "long" else -1
        fill = price * (1 - sign * p.slippage)
        pnl = sign * qty * (fill - entry) - qty * fill * p.fee
        cash += pnl
        trades.append(pnl - open_fee)
        side, qty, entry = None, 0.0, 0.0

    def open_(new_side, price):
        nonlocal cash, side, qty, entry, open_fee
        sign = 1 if new_side == "long" else -1
        fill = price * (1 + sign * p.slippage)
        q = position_size_futures(cash, fill, p, leverage)
        if q > 0:
            open_fee = q * fill * p.fee
            cash -= open_fee
            side, qty, entry = new_side, q, fill

    for i in range(p.ema_slow + 2, len(df)):
        row = df.iloc[i]
        if side:
            sign = 1 if side == "long" else -1
            stop, take = entry * (1 - sign * p.stop_loss_pct), entry * (1 + sign * p.take_profit_pct)
            hit_stop = row.low <= stop if sign == 1 else row.high >= stop
            hit_take = row.high >= take if sign == 1 else row.low <= take
            if hit_stop:
                close(stop)
            elif hit_take:
                close(take)
        c = cross(df.iloc[: i + 1], p)
        if c == "up":
            if side == "short":
                close(row.close)
            if side is None and row.rsi < p.rsi_max_entry:
                open_("long", row.close)
        elif c == "down":
            if side == "long":
                close(row.close)
            if side is None and row.rsi > p.rsi_min_entry:
                open_("short", row.close)
        unreal = (1 if side == "long" else -1) * qty * (row.close - entry) if side else 0.0
        curve.append(cash + unreal)

    last = df.iloc[-1].close
    final = cash + ((1 if side == "long" else -1) * qty * (last - entry) if side else 0.0)
    return final, trades, pd.Series(curve)


def run_signals(df: pd.DataFrame, p: Params, entries, exits, balance: float = config.START_BALANCE):
    """Спот, только покупка: сигналы entries/exits (булевы ряды) по цене закрытия,
    стоп/тейк внутри свечи (стоп приоритетнее), комиссии и проскальзывание как в run()."""
    hi, lo, cl = df.high.values, df.low.values, df.close.values
    en, ex_ = entries.values, exits.values
    cash, qty, entry = balance, 0.0, 0.0
    trades, curve = [], []
    for i in range(len(df)):
        if qty > 0:
            stop, take = entry * (1 - p.stop_loss_pct), entry * (1 + p.take_profit_pct)
            exit_price = stop if lo[i] <= stop else take if hi[i] >= take else None
            if exit_price:
                exit_price *= 1 - p.slippage
                cash += qty * exit_price * (1 - p.fee)
                trades.append(cash_pnl(entry, exit_price, qty, p))
                qty = 0.0
        if qty > 0 and ex_[i]:
            price = cl[i] * (1 - p.slippage)
            cash += qty * price * (1 - p.fee)
            trades.append(cash_pnl(entry, price, qty, p))
            qty = 0.0
        elif qty == 0 and en[i]:
            price = cl[i] * (1 + p.slippage)
            q = min(position_size(cash, price, p), cash / (price * (1 + p.fee)))
            if q > 0:
                cash -= q * price * (1 + p.fee)
                qty, entry = q, price
        curve.append(cash + qty * cl[i])
    return cash + qty * cl[-1], trades, pd.Series(curve)


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
    ap.add_argument("--market", default=config.MARKET, choices=["spot", "futures"])
    a = ap.parse_args()

    if a.synthetic:
        df = synthetic()
    elif a.csv:
        df = pd.read_csv(a.csv, parse_dates=["time"])
    else:
        ex = make_exchange(config.EXCHANGE, market=a.market)
        df = fetch_history(ex, config.SYMBOL, config.TIMEFRAME, a.days)
        df.to_csv("history.csv", index=False)

    p = Params()
    final, trades, curve = (run_futures if a.market == "futures" else run)(df, p)
    report(config.START_BALANCE, final, trades, curve, df)
