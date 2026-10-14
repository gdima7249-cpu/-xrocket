"""Фьючерсный режим (MARKET=futures): long и short на линейных USDT-контрактах.

Логика: пересечение EMA вверх -> long (short закрывается), вниз -> short (long закрывается).
Защита: изолированная маржа, плечо <= 5, стоп-лосс и тейк-профит (на бирже и в боте),
общий стоп-кран. paper-режим моделирует маржу, комиссии и ликвидацию.
"""
import ccxt

import config
from bot import Bot, STATE_FILE, log, notify  # noqa: F401
from config import Params
from strategy import cross, position_size_futures


class FuturesBot(Bot):
    def __init__(self):
        lev, p = config.LEVERAGE, Params()
        if not 1 <= lev <= 5:
            raise SystemExit("LEVERAGE должен быть от 1 до 5")
        if p.stop_loss_pct * lev >= 0.5:
            raise SystemExit("Стоп-лосс слишком широк для такого плеча — риск ликвидации")
        super().__init__()
        if self.mode != "paper":
            self.setup_account()

    # ---------- настройка аккаунта ----------
    def setup_account(self):
        for fn, args in ((self.ex.set_margin_mode, ("isolated", config.SYMBOL)),
                         (self.ex.set_leverage, (config.LEVERAGE, config.SYMBOL))):
            try:
                fn(*args)
            except ccxt.ExchangeError as e:  # часто "not modified" — значит уже стоит
                log.info("Настройка аккаунта: %s", e)

    # ---------- позиция и деньги ----------
    def sync(self):
        """Берём фактическую позицию с биржи (стоп мог сработать на стороне биржи)."""
        if self.mode == "paper":
            return
        pos = next((x for x in self.ex.fetch_positions([config.SYMBOL])
                    if float(x.get("contracts") or 0) > 0), None)
        if pos:
            self.state["side"] = pos["side"]
            self.state["qty"] = float(pos["contracts"]) * float(pos.get("contractSize") or 1)
            self.state["entry"] = float(pos.get("entryPrice") or self.state["entry"])
        else:
            self.state.update(side=None, qty=0.0, entry=0.0)

    def unreal(self, price: float) -> float:
        side = self.state.get("side")
        if not side:
            return 0.0
        sign = 1 if side == "long" else -1
        return sign * self.state["qty"] * (price - self.state["entry"])

    def wallet(self) -> float:
        if self.mode == "paper":
            return self.state["cash"]
        return float(self.ex.fetch_balance()[self.quote]["total"] or 0)

    def free(self) -> float:
        if self.mode == "paper":
            used = (self.state["qty"] * self.state["entry"] / config.LEVERAGE
                    if self.state.get("side") else 0.0)
            return self.state["cash"] - used
        return float(self.ex.fetch_balance()[self.quote]["free"] or 0)

    def equity(self) -> float:
        return self.wallet() + self.unreal(self.price())

    def balances(self):
        return self.free(), self.state["qty"]

    # ---------- сделки ----------
    def open(self, side: str, price: float):
        L, p = config.LEVERAGE, self.p
        equity, free = self.equity(), self.free()
        qty = min(position_size_futures(equity, price, p, L), free * L * 0.95 / price)
        lim = self.ex.market(config.SYMBOL).get("limits", {})
        need = max(((lim.get("cost") or {}).get("min") or 0) / price,
                   (lim.get("amount") or {}).get("min") or 0)
        if qty < need:
            if need * price / L * 1.05 <= free and need * price <= equity * L * p.max_position_pct:
                qty = need * 1.02
                log.info("Минимальный ордер больше расчётного размера — риск на сделку выше обычного")
            else:
                return log.info("Не хватает капитала на минимальный ордер (нужно ~%.2f USDT номинала)",
                                need * price)
        qty = float(self.ex.amount_to_precision(config.SYMBOL, qty))
        if qty <= 0:
            return
        sign = 1 if side == "long" else -1
        if self.mode == "paper":
            self.state["cash"] -= qty * price * p.fee
            self.state.update(side=side, qty=qty, entry=price)
        else:
            order_side = "buy" if side == "long" else "sell"
            sl = price * (1 - sign * p.stop_loss_pct)
            tp = price * (1 + sign * p.take_profit_pct)
            try:   # стоп и тейк на стороне биржи работают, даже если бот выключен
                self.ex.create_order(config.SYMBOL, "market", order_side, qty, None, {
                    "stopLoss": {"triggerPrice": float(self.ex.price_to_precision(config.SYMBOL, sl))},
                    "takeProfit": {"triggerPrice": float(self.ex.price_to_precision(config.SYMBOL, tp))}})
            except ccxt.ExchangeError as e:
                log.warning("Стоп/тейк на бирже не приняты (%s) — ордер без них, стопы ведёт бот", e)
                self.ex.create_order(config.SYMBOL, "market", order_side, qty)
            self.state.update(side=side, qty=qty, entry=price)
            self.sync()
        notify(f"OPEN {side.upper()} {qty} {self.base} @ {price:.2f} (плечо {L:g}x)")

    def close(self, price: float, reason: str):
        side, qty = self.state.get("side"), self.state["qty"]
        if not side or qty <= 0:
            return
        pnl_pct = (price / self.state["entry"] - 1) * 100 * (1 if side == "long" else -1)
        if self.mode == "paper":
            self.state["cash"] += self.unreal(price) - qty * price * self.p.fee
        else:
            self.ex.create_order(config.SYMBOL, "market", "sell" if side == "long" else "buy",
                                 float(self.ex.amount_to_precision(config.SYMBOL, qty)),
                                 None, {"reduceOnly": True})
        self.state.update(side=None, qty=0.0, entry=0.0)
        if self.mode != "paper":
            self.sync()
        notify(f"CLOSE {side.upper()} {qty} {self.base} @ {price:.2f} ({reason}, {pnl_pct:+.2f}% без плеча)")

    def sell(self, price: float, reason: str):   # используется стоп-краном
        self.close(price, reason)

    # ---------- цикл ----------
    def check_stops(self):
        self.sync()
        side = self.state.get("side")
        if not side:
            return
        price, e, p = self.price(), self.state["entry"], self.p
        sign = 1 if side == "long" else -1
        adverse = -sign * (price / e - 1)          # доля движения против позиции
        if self.mode == "paper" and adverse >= 1 / config.LEVERAGE - 0.005:
            self.state["cash"] -= self.state["qty"] * e / config.LEVERAGE
            self.state.update(side=None, qty=0.0, entry=0.0)
            return notify("ЛИКВИДАЦИЯ позиции (paper): потеряна маржа")
        if adverse >= p.stop_loss_pct:
            self.close(price, "stop-loss")
        elif -adverse >= p.take_profit_pct:
            self.close(price, "take-profit")

    def try_open(self, side: str, price: float):
        if self.daily_loss_exceeded():
            return notify("Дневной лимит убытка достигнут — новые позиции приостановлены")
        self.open(side, price)

    def step(self):
        df, p = self.candles(), self.p
        c, cur = cross(df, p), df.iloc[-1]
        self.sync()
        side, price = self.state.get("side"), self.price()
        log.info("price=%.6g rsi=%.1f ema=%.6g/%.6g cross=%s pos=%s",
                 price, cur.rsi, cur.ema_fast, cur.ema_slow, c, side)
        if c == "up":
            if side == "short":
                self.close(price, "signal"); side = None
            if side is None and cur.rsi < p.rsi_max_entry:
                self.try_open("long", price)
        elif c == "down":
            if side == "long":
                self.close(price, "signal"); side = None
            if side is None and cur.rsi > p.rsi_min_entry:
                self.try_open("short", price)


if __name__ == "__main__":
    FuturesBot().run()
