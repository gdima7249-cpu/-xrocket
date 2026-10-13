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


def cross(df: pd.DataFrame, p: Params) -> str | None:
    """Пересечение EMA на последней закрытой свече: 'up', 'down' или None."""
    if len(df) < p.ema_slow + 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    if prev.ema_fast <= prev.ema_slow and cur.ema_fast > cur.ema_slow:
        return "up"
    if prev.ema_fast >= prev.ema_slow and cur.ema_fast < cur.ema_slow:
        return "down"
    return None


def signal(df: pd.DataFrame, p: Params) -> str | None:
    """Сигнал для споты: 'buy' (пересечение вверх, RSI не перегрет), 'sell' (вниз)."""
    c = cross(df, p)
    if c == "up" and df.iloc[-1].rsi < p.rsi_max_entry:
        return "buy"
    if c == "down":
        return "sell"
    return None


def position_size(equity: float, price: float, p: Params) -> float:
    """Размер позиции в базовой валюте: риск = risk_per_trade от капитала."""
    risk_amount = equity * p.risk_per_trade
    qty = risk_amount / (price * p.stop_loss_pct)
    max_qty = equity * p.max_position_pct / price
    return min(qty, max_qty)


def position_size_futures(equity: float, price: float, p: Params, leverage: float) -> float:
    """Размер позиции (в монете): риск = risk_per_trade капитала при стопе stop_loss_pct,
    но не больше max_position_pct * капитал * плечо по номиналу."""
    qty = equity * p.risk_per_trade / (price * p.stop_loss_pct)
    return min(qty, equity * leverage * p.max_position_pct / price)
