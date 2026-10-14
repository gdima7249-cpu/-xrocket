"""Альтернативные стратегии (только покупка, спот). Каждая возвращает два булевых ряда:
entries[i] — открыть позицию по цене закрытия свечи i, exits[i] — закрыть по цене закрытия.
Индикаторы используют только прошлые данные (без заглядывания вперёд)."""
import numpy as np
import pandas as pd

from strategy import rsi


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _valid(df: pd.DataFrame, warmup: int) -> pd.Series:
    return pd.Series(np.arange(len(df)) >= warmup, index=df.index)


def ema_rsi(df, fast, slow, rsi_max=70):
    """Базовая: пересечение EMA вверх (RSI не перегрет) -> вход, вниз -> выход."""
    f, s, r = _ema(df.close, fast), _ema(df.close, slow), rsi(df.close, 14)
    up = (f.shift(1) <= s.shift(1)) & (f > s)
    down = (f.shift(1) >= s.shift(1)) & (f < s)
    v = _valid(df, slow + 2)
    return up & (r < rsi_max) & v, down & v


def trend_pullback(df, trend, rsi_buy, rsi_exit=55):
    """Покупка отката в восходящем тренде: цена выше EMA(trend) и RSI ниже rsi_buy.
    Выход: RSI выше rsi_exit."""
    t, r = _ema(df.close, trend), rsi(df.close, 14)
    v = _valid(df, trend)
    return (df.close > t) & (r < rsi_buy) & v, (r > rsi_exit) & v


def donchian(df, n_in, n_out):
    """Пробой канала: вход при закрытии выше максимума за n_in свечей,
    выход при закрытии ниже минимума за n_out свечей."""
    hi = df.high.rolling(n_in).max().shift(1)
    lo = df.low.rolling(n_out).min().shift(1)
    v = _valid(df, max(n_in, n_out) + 1)
    return (df.close > hi) & v, (df.close < lo) & v


def bollinger(df, n, k):
    """Возврат к среднему: покупка ниже нижней полосы Боллинджера (n, k·σ), выход выше средней."""
    ma, sd = df.close.rolling(n).mean(), df.close.rolling(n).std()
    v = _valid(df, n)
    return (df.close < ma - k * sd) & v, (df.close > ma) & v


STOP_TAKE = [(0.03, 0.06), (0.05, 0.10)]

# (имя, функция, список параметров) — сетка небольшая намеренно: меньше подгонки
GRID = (
    [("ema_rsi", ema_rsi, dict(fast=f, slow=s)) for f, s in [(5, 20), (10, 30), (20, 50)]]
    + [("trend_pullback", trend_pullback, dict(trend=t, rsi_buy=b)) for t in (100, 200) for b in (25, 30, 35)]
    + [("donchian", donchian, dict(n_in=a, n_out=b)) for a in (20, 40, 60) for b in (10, 20)]
    + [("bollinger", bollinger, dict(n=n, k=k)) for n in (20, 40) for k in (2.0, 2.5)]
)
