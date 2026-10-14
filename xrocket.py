"""Адаптер публичного API xRocket Exchange (без ключа) с интерфейсом, как у ccxt.

Подтверждено документацией:
  GET /api/v1/candles?symbol=BTC-USDT&type=1hour&startAt=ISO&endAt=ISO
  -> {"candles": [[time, open, close, high, low, volume], ...]}  (строки)
НЕ подтверждено (проверить при первом запуске): порядок сортировки, точность количества. Ордеры пока не реализованы.
"""
import time
import uuid
from collections import defaultdict
from decimal import ROUND_DOWN, Decimal
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
START_CANDLES = 200  # стартовый размер запроса, при ошибке уменьшается вдвое
MIN_CANDLES = 5


class IntervalTooBig(ccxt.ExchangeError):
    pass


def _iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class XRocketPublic:
    rateLimit = 300

    def __init__(self, testnet: bool = False):
        self.base = BASE["testnet" if testnet else "mainnet"]
        self.http = requests.Session()
        self.max_candles = {}  # timeframe -> подобранный размер запроса
        self.http.headers["Accept"] = "application/json"

    # --- интерфейс, который использует бот ---
    def load_markets(self):
        pass

    def market(self, symbol):
        return {"limits": {"cost": {"min": 0}}}

    def amount_to_precision(self, symbol, amount):
        return f"{int(amount * 1e8) / 1e8:.8f}"

    def parse_timeframe(self, tf):
        return TF_SEC[tf]

    def milliseconds(self):
        return int(time.time() * 1000)

    def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=300):
        """Свечи за период. Если API отвечает "too big interval" — диапазон на
        запрос уменьшается вдвое (размер запоминается), данные докачиваются кусками."""
        limit = limit or 300
        step = TF_SEC[timeframe] * 1000
        end = self.milliseconds()
        start = since if since is not None else end - limit * step
        end = min(end, start + limit * step)
        rows, cur = {}, start
        while cur < end:
            n = self.max_candles.get(timeframe, START_CANDLES)
            chunk_end = min(end, cur + n * step)
            try:
                data = self._get("/api/v1/candles", {
                    "symbol": symbol.split(":")[0].replace("/", "-"), "type": TF[timeframe],
                    "startAt": _iso(cur), "endAt": _iso(chunk_end)})
            except IntervalTooBig:
                if n <= MIN_CANDLES:
                    raise
                self.max_candles[timeframe] = max(MIN_CANDLES, n // 2)
                continue
            for t, o, c, h, l, v in (r[:6] for r in data["candles"]):
                ms = int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() * 1000)
                rows[ms] = [ms, float(o), float(h), float(l), float(c), float(v)]
            cur = chunk_end
            if cur < end:
                time.sleep(self.rateLimit / 1000)
        return [rows[k] for k in sorted(rows)]

    def fetch_ticker(self, symbol):
        """Последняя цена: официальный GET /api/v1/ticker/24h -> tickers[].last
        (так делает пример из руководства xRocket); при сбое — close минутной свечи."""
        sym = symbol.split(":")[0].replace("/", "-")
        try:
            for t in self._get("/api/v1/ticker/24h").get("tickers", []):
                if t.get("symbol") == sym and t.get("last") is not None:
                    return {"last": float(t["last"])}
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError):
            pass
        rows = self.fetch_ohlcv(symbol, "1m", limit=10)
        if not rows:
            raise ccxt.ExchangeError(f"xRocket: нет цены для {symbol}")
        return {"last": rows[-1][4]}

    # --- служебное ---
    def _request(self, method, path, params=None, body=None):
        try:
            r = self.http.request(method, self.base + path, params=params, json=body, timeout=15)
        except requests.RequestException as e:
            raise ccxt.NetworkError(f"xrocket {path}: {e}") from e
        if r.status_code in (401, 403):
            raise ccxt.AuthenticationError(f"xrocket {r.status_code}: проверьте Bearer-токен")
        if r.status_code >= 500:
            raise ccxt.ExchangeNotAvailable(f"xrocket {r.status_code}")
        if r.status_code == 400 and "too_big_interval" in r.text:
            raise IntervalTooBig(f"xrocket 400: {r.text[:200]}")
        if r.status_code not in (200, 201):
            raise ccxt.ExchangeError(f"xrocket {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path, params=None):
        return self._request("GET", path, params=params)


def _step(value: float, step: str) -> str:
    """Округляет вниз до кратного step (строка, как требует API)."""
    s = Decimal(step)
    q = (Decimal(str(value)) / s).to_integral_value(rounding=ROUND_DOWN) * s
    return f"{q.quantize(Decimal(1).scaleb(s.as_tuple().exponent)):f}"


class XRocketExchange(XRocketPublic):
    """Спот на xRocket Exchange API (приватные методы, Bearer-токен).

    Подтверждено документацией: POST /api/v1/orders (market: symbol, side,
    type, size [базовая монета], timeInForce=IOC), GET /api/v1/accounts/trading/balances,
    GET /api/v1/symbols/{symbol}, GET /api/v1/orders/history.
    Ордер: POST /api/v1/orders, type=market, size (базовая монета), timeInForce=IOC — как в руководстве.
    Не подтверждено: путь «Получить ордер» (исполнение проверяем по истории ордеров).
    """
    TERMINAL = {"completed", "cancelled", "rejected", "expired"}

    def __init__(self, token: str, testnet: bool = False):
        super().__init__(testnet)
        self.http.headers["Authorization"] = f"Bearer {token}"
        self._info_cache = {}

    @staticmethod
    def _sym(symbol):
        return symbol.split(":")[0].replace("/", "-")

    def _info(self, symbol):
        sym = self._sym(symbol)
        if sym not in self._info_cache:
            d = self._get(f"/api/v1/symbols/{sym}")
            if "symbols" in d:                      # на случай ответа-списка
                d = d["symbols"][0]
            self._info_cache[sym] = d
        return self._info_cache[sym]

    def market(self, symbol):
        i = self._info(symbol)
        return {"limits": {"amount": {"min": float(i["baseMinSize"])},
                           "cost": {"min": float(i["quoteMinSize"])}},
                "maker": None, "taker": None, "info": i}

    def amount_to_precision(self, symbol, amount):
        return _step(amount, self._info(symbol)["baseIncrement"])

    def price_to_precision(self, symbol, price):
        return _step(price, self._info(symbol)["priceIncrement"])

    def fetch_balance(self):
        """Торговый баланс (не основной!): {asset: {free, total, used}}."""
        out = defaultdict(lambda: {"free": 0.0, "total": 0.0, "used": 0.0})
        for b in self._get("/api/v1/accounts/trading/balances")["balances"]:
            out[b["asset"]] = {"free": float(b["available"]), "total": float(b["balance"]),
                               "used": float(b["holds"])}
        return out

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        """Рыночный ордер, как в руководстве xRocket: type=market, size в базовой монете, IOC."""
        if type != "market":
            raise ccxt.NotSupported("xrocket: поддерживаются только рыночные ордера")
        order = self._request("POST", "/api/v1/orders", body={
            "symbol": self._sym(symbol), "side": side, "type": "market",
            "size": self.amount_to_precision(symbol, float(amount)), "timeInForce": "IOC"})
        fill = self._await_fill(symbol, order)
        res = {"id": order.get("id"), "info": order, "filled": None, "average": None,
               "fill_checked": False}
        if fill:
            size, funds = float(fill.get("dealSize") or 0), float(fill.get("dealFunds") or 0)
            res.update(filled=size, average=funds / size if size else None, fill_checked=True)
        return res

    def create_market_sell_order(self, symbol, amount):
        return self.create_order(symbol, "market", "sell", amount)

    def _await_fill(self, symbol, order):
        """Итог ордера: сразу из ответа или по истории ордеров (до ~8 секунд), поиск по id."""
        for attempt in range(9):
            if order.get("status") in self.TERMINAL and "dealSize" in order:
                return order
            if attempt:
                time.sleep(1.0)
            now = self.milliseconds()
            hist = self._get("/api/v1/orders/history", {
                "symbol": self._sym(symbol), "startAt": _iso(now - 10 * 60 * 1000),
                "endAt": _iso(now + 60 * 1000), "currentPage": 1, "pageSize": 20,
                "hideCanceled": "false"})
            order = next((o for o in hist.get("orders", []) if o.get("id") == order.get("id")), order)
        return order if order.get("status") in self.TERMINAL else None
