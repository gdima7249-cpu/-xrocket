"""Сканер просадок: следит за ВСЕМИ парами к USDT на xRocket, покупает резкое падение, продаёт на отскоке.

Режимы (MODE в .env): paper (виртуальные деньги, реальные цены) | testnet | live.
Раз в час (после закрытия часовой свечи) для каждой пары: если цена ниже максимума за SCAN_LOOKBACK свечей
на SCAN_DROP (и, при SCAN_CONFIRM=1, свеча зелёная) — покупка на SCAN_POSITION_PCT капитала (до SCAN_MAX_POSITIONS).
Каждую минуту: выход по тейк-профиту / стоп-лоссу / таймауту. Общий стоп-кран MAX_TOTAL_LOSS_PCT как у основного бота.
"""
import json
import os
import time
from datetime import datetime, timezone

import ccxt

import config
from bot import log, notify
from config import Params
from exchanges import make_exchange

STATE_FILE = "scanner_state.json"


class Scanner:
    def __init__(self):
        if config.EXCHANGE != "xrocket":
            raise SystemExit("Сканер пока работает только с xRocket (EXCHANGE=xrocket).")
        self.mode, self.p = config.MODE, Params()
        self.ex = make_exchange("xrocket", self.mode, config.API_KEY, config.API_SECRET)
        if self.mode == "live" and input("MODE=live: реальные деньги! Введите YES для продолжения: ") != "YES":
            raise SystemExit("Отменено")
        self.universe = [s["symbol"] for s in self.ex.list_symbols()
                         if s["symbol"].endswith("-USDT") and s.get("enableTrading", True)]
        log.info("Пар к USDT под наблюдением: %d", len(self.universe))
        self.s = self.load()
        if "start_equity" not in self.s:
            self.s["start_equity"] = self.equity(self.ex.fetch_tickers())

    # ---------- состояние ----------
    def load(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                st = json.load(f)
            if st.get("start_balance") == config.START_BALANCE:
                return st
            log.info("PAPER_BALANCE изменён — состояние сканера сброшено")
        return {"cash": config.START_BALANCE, "pos": {}, "cool": {}, "start_balance": config.START_BALANCE}

    def save(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.s, f)

    # ---------- деньги ----------
    def free_quote(self):
        if self.mode == "paper":
            return self.s["cash"]
        return float(self.ex.fetch_balance()["USDT"]["free"])

    def equity(self, prices):
        cash = self.s["cash"] if self.mode == "paper" else float(self.ex.fetch_balance()["USDT"]["total"])
        return cash + sum(v["qty"] * prices.get(sym, v["entry"]) for sym, v in self.s["pos"].items())

    # ---------- сделки ----------
    def buy(self, sym, price, prices):
        eq = self.equity(prices)
        value = min(eq * config.SCAN_POSITION_PCT, self.free_quote() * 0.98)
        lim = self.ex.market(sym)["limits"]
        min_cost = (lim.get("cost") or {}).get("min") or 0
        qty = float(self.ex.amount_to_precision(sym, value / price))
        if qty <= 0 or qty * price < max(min_cost, 0) or qty < ((lim.get("amount") or {}).get("min") or 0):
            return log.info("%s: сумма %.4f USDT меньше минимального ордера", sym, value)
        entry = price * (1 + self.p.slippage)
        if self.mode == "paper":
            self.s["cash"] -= qty * entry * (1 + self.p.fee)
        else:
            o = self.ex.create_order(sym, "market", "buy", qty, price)
            if o.get("fill_checked"):
                if not o["filled"]:
                    return log.info("%s: ордер не исполнился", sym)
                qty = o["filled"]
            entry = o.get("average") or price
        self.s["pos"][sym] = {"qty": qty, "entry": entry, "t": time.time()}
        notify(f"BUY {sym} {qty} @ {entry:.6g}")

    def sell(self, sym, price, reason):
        pos = self.s["pos"][sym]
        qty = pos["qty"]
        if self.mode == "paper":
            self.s["cash"] += qty * price * (1 - self.p.slippage) * (1 - self.p.fee)
        else:
            free = float(self.ex.fetch_balance()[sym.split("-")[0]]["free"])
            q = float(self.ex.amount_to_precision(sym, min(qty, free)))
            o = self.ex.create_market_sell_order(sym, q) if q > 0 else {}
            if o.get("fill_checked") and o["filled"] < q * 0.999:
                pos["qty"] = max(0.0, qty - o["filled"])
                return notify(f"{sym}: продано частично ({o['filled']} из {q}), остаток продам на следующем шаге")
        pnl = (price / pos["entry"] - 1) * 100
        del self.s["pos"][sym]
        self.s["cool"][sym] = time.time()
        notify(f"SELL {sym} {qty} @ {price:.6g} ({reason}, {pnl:+.2f}%)")

    # ---------- логика ----------
    def manage(self, prices):
        """Выходы по тейку / стопу / таймауту (каждую минуту)."""
        for sym in list(self.s["pos"]):
            if sym not in prices:
                continue
            v, price = self.s["pos"][sym], prices[sym]
            r = price / v["entry"] - 1
            hours = (time.time() - v["t"]) / 3600
            if r >= config.SCAN_TAKE:
                self.sell(sym, price, "take-profit")
            elif r <= -config.SCAN_STOP:
                self.sell(sym, price, "stop-loss")
            elif hours >= config.SCAN_MAX_HOLD_H:
                self.sell(sym, price, "timeout")

    def scan(self, prices):
        """Поиск просадок на закрытых часовых свечах (раз в час)."""
        slots = config.SCAN_MAX_POSITIONS - len(self.s["pos"])
        if slots <= 0:
            return log.info("Сканер: все %d позиций заняты", config.SCAN_MAX_POSITIONS)
        found = []
        for sym in self.universe:
            if sym in self.s["pos"] or sym not in prices:
                continue
            if time.time() - self.s["cool"].get(sym, 0) < config.SCAN_LOOKBACK * 3600:
                continue            # недавно уже торговали этой парой
            try:
                c = self.ex.fetch_ohlcv(sym, "1h", limit=config.SCAN_LOOKBACK + 3)[:-1]   # без незакрытой свечи
            except ccxt.ExchangeError:
                continue
            if len(c) < config.SCAN_LOOKBACK:
                continue
            w = c[-config.SCAN_LOOKBACK:]
            peak, last = max(x[4] for x in w), w[-1]
            drop = last[4] / peak - 1
            if drop <= -config.SCAN_DROP and (not config.SCAN_CONFIRM or last[4] > last[1]):
                found.append((drop, sym))
        found.sort()
        log.info("Сканер: просевших пар найдено %d, свободных мест %d", len(found), slots)
        for drop, sym in found[:slots]:
            log.info("%s: просадка %.1f%% -> покупка", sym, drop * 100)
            self.buy(sym, prices[sym], prices)

    def kill_switch(self, prices):
        if self.s.get("halted"):
            return True
        floor = config.STOP_EQUITY if config.STOP_EQUITY is not None else \
            self.s["start_equity"] * (1 - config.MAX_TOTAL_LOSS_PCT)
        eq = self.equity(prices)
        if eq > floor:
            return False
        for sym in list(self.s["pos"]):
            self.sell(sym, prices.get(sym, self.s["pos"][sym]["entry"]), "kill-switch")
        self.s["halted"] = True
        self.save()
        notify(f"СТОП: капитал {eq:.4f} USDT достиг лимита {floor:.4f}. Сканер остановлен. Для сброса удалите {STATE_FILE}.")
        return True

    def run(self):
        notify(f"Сканер запущен: mode={self.mode}, пар {len(self.universe)}, падение {config.SCAN_DROP*100:.0f}% "
               f"за {config.SCAN_LOOKBACK}ч, тейк {config.SCAN_TAKE*100:.0f}%, стоп {config.SCAN_STOP*100:.0f}%")
        last_hour = None
        while True:
            try:
                prices = self.ex.fetch_tickers()
                if self.kill_switch(prices):
                    return log.info("Торговля остановлена. Для нового запуска удалите %s", STATE_FILE)
                self.manage(prices)
                hour = int(time.time() // 3600)
                if hour != last_hour:
                    last_hour = hour
                    time.sleep(5)           # дать бирже закрыть свечу
                    self.scan(prices)
                    log.info("Позиций: %d, капитал ≈ %.4f USDT", len(self.s["pos"]), self.equity(prices))
                self.save()
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                log.warning("Сеть/биржа: %s", e)
            except Exception as e:  # noqa: BLE001
                notify(f"Ошибка сканера: {e!r}")
            time.sleep(60)


if __name__ == "__main__":
    Scanner().run()
