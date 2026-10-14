"""Торговый бот. Режимы (MODE в .env): paper | demo | testnet | live.

paper   — реальные цены, виртуальный баланс, ордера не отправляются
demo    — демо-торговля биржи с виртуальными деньгами (Bybit Demo, нужны демо-ключи)
testnet — ордера на песочнице биржи (нужны testnet-ключи)
live    — РЕАЛЬНЫЕ деньги
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import requests

import config
from config import Params
from exchanges import make_exchange
from strategy import add_indicators, position_size, signal_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")
STATE_FILE = "state.json"


def notify(text: str):
    log.info(text)
    if config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        except requests.RequestException as e:
            log.warning("Telegram: %s", e)


class Bot:
    def __init__(self):
        self.p = Params()
        self.mode = config.MODE
        self.ex = make_exchange(config.EXCHANGE, self.mode, config.API_KEY, config.API_SECRET,
                                 config.MARKET)
        if self.mode == "live":
            if input("MODE=live: реальные деньги! Введите YES для продолжения: ") != "YES":
                raise SystemExit("Отменено")
        self.ex.load_markets()
        self.base, self.quote = config.SYMBOL.split(":")[0].split("/")
        self.state = self.load_state()
        if "start_equity" not in self.state:
            self.state["start_equity"] = (config.START_BALANCE if self.mode == "paper"
                                          else self.equity())

    # ---------- состояние ----------
    def load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                st = json.load(f)
            # Сменили PAPER_BALANCE -> начинаем с чистого виртуального счёта
            if (st.get("start_balance") == config.START_BALANCE
                    and st.get("market", "spot") == config.MARKET):
                return st
            log.info("PAPER_BALANCE или MARKET изменён — виртуальный счёт сброшен")
        return {"cash": config.START_BALANCE, "qty": 0.0, "entry": 0.0,
                "day": "", "day_start_equity": config.START_BALANCE,
                "start_balance": config.START_BALANCE, "market": config.MARKET}

    def save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f)

    # ---------- биржа ----------
    def price(self) -> float:
        return self.ex.fetch_ticker(config.SYMBOL)["last"]

    def balances(self):
        if self.mode == "paper":
            return self.state["cash"], self.state["qty"]
        b = self.ex.fetch_balance()
        # позиция бота — из state (чужие монеты на кошельке не трогаем)
        return b[self.quote]["free"], min(self.state["qty"], b[self.base]["free"])

    def equity(self) -> float:
        cash, qty = self.balances()
        return cash + qty * self.price()

    def candles(self) -> pd.DataFrame:
        raw = self.ex.fetch_ohlcv(config.SYMBOL, config.TIMEFRAME, limit=300)
        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume"])
        df = df.iloc[:-1]  # последняя свеча ещё не закрыта — отбрасываем
        return add_indicators(df, self.p)

    # ---------- сделки ----------
    def buy(self, price: float):
        cash, _ = self.balances()
        qty = position_size(self.equity(), price, self.p)
        qty = min(qty, cash / (price * (1 + self.p.fee)))
        qty = float(self.ex.amount_to_precision(config.SYMBOL, qty))
        market = self.ex.market(config.SYMBOL)
        min_cost = (market.get("limits", {}).get("cost", {}) or {}).get("min") or 0
        if qty <= 0 or qty * price < min_cost:
            return log.info("Сумма слишком мала для ордера (мин. %s)", min_cost)
        if self.mode == "paper":
            self.state["cash"] -= qty * price * (1 + self.p.fee)
            self.state["qty"], self.state["entry"] = qty, price
        else:
            # цена нужна ccxt, чтобы пересчитать количество в сумму (Bybit spot и др.)
            order = self.ex.create_order(config.SYMBOL, "market", "buy", qty, price)
            if order.get("fill_checked"):          # биржа подтвердила фактическое исполнение
                if not order["filled"]:
                    return log.info("Ордер не исполнился (IOC) — позиция не открыта")
                qty = order["filled"]
            self.state["entry"] = order.get("average") or price
            self.state["qty"] = qty
        notify(f"BUY {qty} {self.base} @ {price:.2f}")

    def sell(self, price: float, reason: str):
        _, qty = self.balances()
        qty = float(self.ex.amount_to_precision(config.SYMBOL, qty))
        if qty <= 0:
            return
        if self.mode == "paper":
            self.state["cash"] += qty * price * (1 - self.p.fee)
            self.state["qty"] = 0.0
        else:
            order = self.ex.create_market_sell_order(config.SYMBOL, qty) or {}
            if order.get("fill_checked") and order["filled"] < qty * 0.999:
                self.state["qty"] = max(0.0, self.state["qty"] - order["filled"])
                return notify(f"Продажа исполнена на {order['filled']} из {qty} {self.base} "
                              "(IOC) — остаток будет продан на следующем шаге")
            self.state["qty"] = 0.0
        pnl = (price / self.state["entry"] - 1) * 100
        notify(f"SELL {qty} {self.base} @ {price:.2f} ({reason}, {pnl:+.2f}%)")
        self.state["entry"] = 0.0

    # ---------- дневной лимит убытка ----------
    def daily_loss_exceeded(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        eq = self.equity()
        if self.state["day"] != today:
            self.state.update(day=today, day_start_equity=eq)
        return eq < self.state["day_start_equity"] * (1 - self.p.max_daily_loss_pct)

    # ---------- общий стоп-кран ----------
    def equity_floor(self) -> float:
        if config.STOP_EQUITY is not None:
            return config.STOP_EQUITY
        return self.state["start_equity"] * (1 - config.MAX_TOTAL_LOSS_PCT)

    def kill_switch(self) -> bool:
        """True, если торговля остановлена. Остановка необратима до удаления state.json."""
        if self.state.get("halted"):
            return True
        eq, floor = self.equity(), self.equity_floor()
        if eq > floor:
            return False
        if self.balances()[1] > 0:
            self.sell(self.price(), "kill-switch")
        self.state["halted"] = True
        self.save_state()
        notify(f"СТОП: капитал {eq:.4f} {self.quote} достиг лимита {floor:.4f} "
               f"(старт {self.state['start_equity']:.4f}). Бот остановлен. "
               f"Чтобы начать заново, удалите {STATE_FILE}.")
        return True

    # ---------- цикл ----------
    def check_stops(self):
        if self.state["qty"] <= 0 and self.mode == "paper":
            return
        _, qty = self.balances()
        if qty <= 0 or not self.state["entry"]:
            return
        price, e = self.price(), self.state["entry"]
        if price <= e * (1 - self.p.stop_loss_pct):
            self.sell(price, "stop-loss")
        elif price >= e * (1 + self.p.take_profit_pct):
            self.sell(price, "take-profit")

    def step(self):
        df = self.candles()
        sig = signal_for(df, self.p)
        _, qty = self.balances()
        price = self.price()
        log.info("price=%.6g rsi=%.1f ema=%.6g/%.6g signal=%s pos=%s",
                 price, df.iloc[-1].rsi, df.iloc[-1].ema_fast, df.iloc[-1].ema_slow, sig, qty)
        if sig == "buy" and qty == 0:
            if self.daily_loss_exceeded():
                notify("Дневной лимит убытка достигнут — покупки приостановлены")
            else:
                self.buy(price)
        elif sig == "sell" and qty > 0:
            self.sell(price, "signal")

    def run(self):
        notify(f"Бот запущен: {config.SYMBOL} {config.TIMEFRAME} mode={self.mode} strategy={config.STRATEGY}")
        tf_sec = self.ex.parse_timeframe(config.TIMEFRAME)
        last_candle = None
        while True:
            try:
                if self.kill_switch():
                    return log.info("Торговля остановлена (стоп-кран). "
                                    "Для нового запуска удалите %s", STATE_FILE)
                self.check_stops()
                cur = int(time.time() // tf_sec)
                if cur != last_candle:      # новая свеча закрылась
                    last_candle = cur
                    time.sleep(3)           # дать бирже закрыть свечу
                    self.step()
                self.save_state()
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                log.warning("Сеть/биржа: %s", e)
            except Exception as e:  # noqa: BLE001
                notify(f"Ошибка: {e!r}")
            time.sleep(30)


if __name__ == "__main__":
    if config.MARKET == "futures":
        from futures_bot import FuturesBot
        FuturesBot().run()
    else:
        Bot().run()
