import time

import numpy as np
import pandas as pd

COLUMNS = ["time", "open", "high", "low", "close", "volume"]


def fetch_history(exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Скачивает историю свечей постранично (публичные данные, ключи не нужны)."""
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - days * 86400 * 1000
    rows = []
    while since < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + ms_per_candle
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("time")
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.reset_index(drop=True)


def synthetic(n: int = 5000, seed: int = 1) -> pd.DataFrame:
    """Случайные данные для проверки кода без интернета (НЕ для оценки стратегии)."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.01, n)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    t = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"time": t, "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(1, 100, n)})
