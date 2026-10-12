"""Адаптер публичного API xRocket Exchange (без ключа) с интерфейсом, как у ccxt.

Подтверждено документацией:
  GET /api/v1/candles?symbol=BTC-USDT&type=1hour&startAt=ISO&endAt=ISO
  -> {"candles": [[time, open, close, high, low, volume], ...]}  (строки)
НЕ подтверждено (проверить при первом запуске): лимит свечей на запрос,
порядок сортировки, точность количества. Ордеры пока не реализованы.
"""
import time
from datetime import datetime, timedelta, timezone

import ccxt
import requests

BASE = {
    "mainnet": "https://exchange.api.xrocket.exchange",
    "testnet": "https://exchange.api.testnet.xrocket.exchange",
}
TF = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1hour",
      "2h": "2hour", "4h": "4hour", "8h": "8hour", "12h": "12hour",
      "1d": "1day", "1w": "1week", "1M": "1month"}
TF_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
          "4h": 14400, "8h": 28800, "12h": 43200, "1d": 86400, "1w": 604800, "1M": 2592000}
MAX_CANDLES = 500  # осторожный лимит на один запрос


def _iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class XRocketPublic:
    rateLimit = 300

    def __init__(self, testnet: bool = False):
        self.base = BASE["testnet" if testnet else "mainnet"]
        self.http = requests.Session()
        self.http.headers["Accept"] = "application/json"

    # --- интерфейс, который использует бот ---
    def load_markets(self):
        pass

    def market(self, symbol):
        return {"limits": {"cost": {"min": 0}}}

    def amount_to_precision(self, symbol, amount):
        return f"{int(amount * 1e6) / 1e6:.6f}"

    def parse_timeframe(self, tf):
        return TF_SEC[tf]

    def milliseconds(self):
        return int(time.time() * 1000)

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=300):
        limit = min(limit or 300, MAX_CANDLES)
        step = TF_SEC[timeframe] * 1000
        end = self.milliseconds()
        start = since if since is not None else end - limit * step
        end = min(end, start + limit * step)
        data = self._get("/api/v1/candles", {
            "symbol": symbol.replace("/", "-"), "type": TF[timeframe],
            "startAt": _iso(start), "endAt": _iso(end)})
        rows = []
        for t, o, c, h, l, v in (r[:6] for r in data["candles"]):
            ms = int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() * 1000)
            rows.append([ms, float(o), float(h), float(l), float(c), float(v)])
        return sorted(rows)

    def fetch_ticker(self, symbol):
        # Схема тикера не подтверждена -> берём close последней минутной свечи
        rows = self.fetch_ohlcv(symbol, "1m", limit=10)
        if not rows:
            raise ccxt.ExchangeError(f"xRocket: нет свечей для {symbol}")
        return {"last": rows[-1][4]}

    # --- служебное ---
    def _get(self, path, params):
        try:
            r = self.http.get(self.base + path, params=params, timeout=15)
        except requests.RequestException as e:
            raise ccxt.NetworkError(f"xrocket {path}: {e}") from e
        if r.status_code >= 500:
            raise ccxt.ExchangeNotAvailable(f"xrocket {r.status_code}")
        if r.status_code != 200:
            raise ccxt.ExchangeError(f"xrocket {r.status_code}: {r.text[:200]}")
        return r.json()
