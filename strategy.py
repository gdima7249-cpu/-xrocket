import numpy as np
import pandas as pd

from config import Params


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=p.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=p.ema_slow, adjust=False).mean()
    df["rsi"] = rsi(df["close"], p.rsi_period)
    return df


def signal(df: pd.DataFrame, p: Params) -> str | None:
    """Сигнал по последней ЗАКРЫТОЙ свече: 'buy', 'sell' или None.

    buy  — быстрая EMA пересекла медленную снизу вверх и RSI не перегрет
    sell — быстрая EMA пересекла медленную сверху вниз
    """
    if len(df) < p.ema_slow + 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    crossed_up = prev.ema_fast <= prev.ema_slow and cur.ema_fast > cur.ema_slow
    crossed_down = prev.ema_fast >= prev.ema_slow and cur.ema_fast < cur.ema_slow
    if crossed_up and cur.rsi < p.rsi_max_entry:
        return "buy"
    if crossed_down:
        return "sell"
    return None


def position_size(equity: float, price: float, p: Params) -> float:
    """Размер позиции в базовой валюте: риск = risk_per_trade от капитала."""
    risk_amount = equity * p.risk_per_trade
    qty = risk_amount / (price * p.stop_loss_pct)
    max_qty = equity * p.max_position_pct / price
    return min(qty, max_qty)
