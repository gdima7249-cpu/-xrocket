"""Виртуальная торговля на НЕСКОЛЬКИХ биржах по РЕАЛЬНЫМ данным (ключи не нужны, деньги не рискуются).

Одни и те же сигналы стратегий (dip, donchian, ema_rsi, trend_pullback, bollinger) исполняются «виртуально» на каждой бирже:
рыночный ордер моделируется по НАСТОЯЩЕМУ стакану заявок (спред и проскальзывание считаются как на самом деле), комиссия —
настоящая тейкер-комиссия биржи. xRocket: стакан недоступен -> модель (комиссия 0.3%, проскальзывание 0.25%, измерено на testnet).
Итог показывает не «работает ли на тесте», а сколько стоит вход/выход на каждой площадке и что остаётся после этого.

    python multi_paper.py            # постоянная работа (отчёт раз в час)
    python multi_paper.py --once     # один проход сигналов и выход
    python multi_paper.py --stats    # сводка из файла

Настройки (.env): MULTI_EXCHANGES=binance,okx,bybit,xrocket  MULTI_SYMBOLS=BTC/USDT,ETH/USDT,...  MULTI_START=1000
MULTI_POSITION_PCT=0.1 (доля капитала на позицию)  MULTI_MAX_POSITIONS=8
"""
import argparse
import json
import os
import time
from datetime import datetime

import ccxt
import pandas as pd

import config  # noqa: F401  (подгружает .env)
from data import COLUMNS
from exchanges import make_exchange
from radar import STRATS

STATE_FILE = os.getenv("MULTI_STATE", "multi_state.json")
EXCHANGES = [e.strip() for e in os.getenv("MULTI_EXCHANGES", "binance,okx,bybit,xrocket").split(",") if e.strip()]
SYMBOLS = [s.strip() for s in os.getenv(
    "MULTI_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,BNB/USDT,DOGE/USDT,ADA/USDT,LINK/USDT").split(",") if s.strip()]
START = float(os.getenv("MULTI_START", "1000"))
POS_PCT = float(os.getenv("MULTI_POSITION_PCT", "0.1"))
MAX_POS = int(os.getenv("MULTI_MAX_POSITIONS", "8"))
MODEL_FEE, MODEL_SLIP = 0.003, 0.0025          # xRocket: измерено на testnet


class Venue:
    """Одна биржа: данные, комиссия и моделирование исполнения по стакану."""

    def __init__(self, name, ex=None):
        self.name = name
        self.modeled = name == "xrocket"
        self.ex = ex or (make_exchange("xrocket", "paper") if self.modeled
                         else getattr(ccxt, name)({"enableRateLimit": True}))
        if not self.modeled:
            self.ex.load_markets()
        self.symbols = [s for s in SYMBOLS if self.modeled or s in self.ex.markets]

    def fee(self, sym):
        if self.modeled:
            return MODEL_FEE
        return (self.ex.markets.get(sym) or {}).get("taker") or 0.001

    def candles(self, sym, tf):
        rows = self.ex.fetch_ohlcv(sym, tf, limit=260)[:-1]           # без незакрытой свечи
        return pd.DataFrame(rows, columns=COLUMNS)

    def marks(self):
        """Цена, по которой можно продать прямо сейчас (bid): {symbol: price}."""
        if self.modeled:
            last = self.ex.fetch_tickers()
            return {s: last[s.replace("/", "-")] * (1 - MODEL_SLIP) for s in self.symbols if s.replace("/", "-") in last}
        t = self.ex.fetch_tickers(self.symbols)
        return {s: (t[s].get("bid") or t[s].get("last")) for s in self.symbols if s in t and (t[s].get("bid") or t[s].get("last"))}

    def buy_fill(self, sym, quote):
        """Купить на `quote` USDT: (количество до комиссии, средняя цена, середина спреда) или None."""
        if self.modeled:
            last = self.ex.fetch_ticker(sym)["last"]
            price = last * (1 + MODEL_SLIP)
            return quote / price, price, last
        book = self.ex.fetch_order_book(sym, 50)
        mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
        left, qty = quote, 0.0
        for price, amount in book["asks"]:
            take = min(left / price, amount)
            qty, left = qty + take, left - take * price
            if left <= 1e-9:
                return qty, quote / qty, mid
        return None                                          # стакана не хватило

    def sell_fill(self, sym, qty):
        """Продать qty: (выручка до комиссии, средняя цена, середина спреда) или None."""
        if self.modeled:
            last = self.ex.fetch_ticker(sym)["last"]
            price = last * (1 - MODEL_SLIP)
            return qty * price, price, last
        book = self.ex.fetch_order_book(sym, 50)
        mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
        left, got = qty, 0.0
        for price, amount in book["bids"]:
            take = min(left, amount)
            got, left = got + take * price, left - take
            if left <= 1e-12:
                return got, got / qty, mid
        return None


class Multi:
    def __init__(self, venues=None):
        self.venues = venues or self.connect()
        self.s = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
        for v in self.venues:
            self.s.setdefault(v.name, {"cash": START, "open": {}, "closed": [], "arb": {}, "rt_costs": []})

    @staticmethod
    def connect():
        out = []
        for name in EXCHANGES:
            try:
                v = Venue(name)
                print(f"{name}: подключено, пар {len(v.symbols)} ({'модель' if v.modeled else 'реальный стакан'})")
                out.append(v)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: пропущена ({str(e)[:100]})")
        if not out:
            raise SystemExit("Ни одна биржа не подключилась.")
        return out

    def save(self):
        json.dump(self.s, open(STATE_FILE, "w"))

    def equity(self, v, marks):
        a = self.s[v.name]
        return a["cash"] + sum(p["qty"] * marks.get(k.split("|")[1], p["avg"]) for k, p in a["open"].items())

    # ---------- сделки ----------
    def open_trade(self, v, st, sym, marks):
        a = self.s[v.name]
        if len(a["open"]) >= MAX_POS or f"{st}|{sym}" in a["open"]:
            return False
        cost = min(self.equity(v, marks) * POS_PCT, a["cash"] * 0.98)
        if cost < 10:
            return False
        fill = v.buy_fill(sym, cost)
        if not fill:
            return False
        qty_gross, avg, mid = fill
        fee = v.fee(sym)
        a["cash"] -= cost
        a["open"][f"{st}|{sym}"] = {"avg": avg, "mid": mid, "qty": qty_gross * (1 - fee), "cost": cost, "fee": fee, "t": time.time()}
        return True

    def close_trade(self, v, key, reason):
        a = self.s[v.name]
        p = a["open"][key]
        st, sym = key.split("|")
        fill = v.sell_fill(sym, p["qty"])
        if not fill:
            return None
        gross, avg, mid = fill
        net = gross * (1 - v.fee(sym))
        a["cash"] += net
        del a["open"][key]
        ret = net / p["cost"] - 1
        # измеренная стоимость круга: спред/проскальзывание на входе и выходе + две комиссии
        a["rt_costs"].append((p["avg"] / p["mid"] - 1) + (1 - avg / mid) + p["fee"] + v.fee(sym))
        a["closed"].append({"s": st, "sym": sym, "ret": ret, "reason": reason, "t": time.time(), "t0": p["t"]})
        return ret

    def manage(self, v, marks):
        a = self.s[v.name]
        for key in list(a["open"]):
            st, sym = key.split("|")
            if sym not in marks:
                continue
            p, cfg = a["open"][key], STRATS[st]
            r = marks[sym] / p["avg"] - 1
            if r >= cfg["tp"]:
                self.close_trade(v, key, "take-profit")
            elif r <= -cfg["sl"]:
                self.close_trade(v, key, "stop-loss")
            elif (time.time() - p["t"]) / 3600 >= cfg["hold"]:
                self.close_trade(v, key, "timeout")

    def scan(self, v, marks, tfs):
        found = []
        a = self.s[v.name]
        for sym in v.symbols:
            if sym not in marks:
                continue
            frames = {}
            for st, cfg in STRATS.items():
                if cfg["tf"] not in tfs:
                    continue
                try:
                    if cfg["tf"] not in frames:
                        frames[cfg["tf"]] = v.candles(sym, cfg["tf"])
                    d = frames[cfg["tf"]]
                except ccxt.BaseError:
                    break
                if len(d) < 60:
                    continue
                en, ex_ = cfg["fn"](d)
                key = f"{st}|{sym}"
                if key in a["open"] and bool(ex_.iloc[-1]):
                    self.close_trade(v, key, "signal")
                elif key not in a["open"] and bool(en.iloc[-1]) and self.open_trade(v, st, sym, marks):
                    found.append((st, sym))
        return found

    # ---------- отчёт ----------
    def report(self, marks_by):
        print(f"\n{'=' * 82}\nВИРТУАЛЬНАЯ ТОРГОВЛЯ НА НЕСКОЛЬКИХ БИРЖАХ {datetime.now():%Y-%m-%d %H:%M}   (старт {START:.0f} USDT на бирже)")
        print(f"  {'биржа':<10}{'данные':<9}{'капитал':>10}{'изм.%':>8}{'закрыто':>9}{'открыто':>9}{'ср.% сделка':>13}{'стоимость круга %':>18}")
        for v in self.venues:
            a = self.s[v.name]
            eq = self.equity(v, marks_by.get(v.name, {}))
            rets = [c["ret"] * 100 for c in a["closed"]]
            rt = a["rt_costs"]
            print(f"  {v.name:<10}{'модель' if v.modeled else 'стакан':<9}{eq:>10.2f}{(eq / START - 1) * 100:>+8.2f}{len(rets):>9}{len(a['open']):>9}"
                  f"{(sum(rets) / len(rets) if rets else 0):>+13.2f}{(sum(rt) / len(rt) * 100 if rt else 0):>18.2f}")
        print("«Стоимость круга» — сколько в среднем отнимает вход+выход (спред, проскальзывание, 2 комиссии) на этой бирже.")
        print("Ориентир: если стоимость круга около 0.2% и выше, а сделки в среднем дают меньше, стратегия не окупает биржу.")


def run(once=False):
    m = Multi()
    last = {"1h": None, "4h": None}
    marks_by = {}
    if not once:
        print("Работает. Подробный отчёт раз в час; остановка: Ctrl+C.")
    while True:
        now = time.time()
        tfs = [tf for tf, sec in (("1h", 3600), ("4h", 14400)) if last[tf] != int(now // sec)]
        for v in m.venues:
            try:
                marks_by[v.name] = v.marks()
                m.manage(v, marks_by[v.name])
                if tfs:
                    m.scan(v, marks_by[v.name], tfs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.ExchangeError) as e:
                print(f"{v.name}: {str(e)[:100]}")
        if tfs:
            for tf, sec in (("1h", 3600), ("4h", 14400)):
                if tf in tfs:
                    last[tf] = int(now // sec)
            m.report(marks_by)
        m.save()
        if once:
            return
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.stats:
        if not os.path.exists(STATE_FILE):
            raise SystemExit("Файла состояния ещё нет.")
        s = json.load(open(STATE_FILE))
        from radar import stats_text
        for name, st in s.items():
            print(f"\n=== {name} === (кэш {st['cash']:.2f}, открыто {len(st['open'])})")
            print(stats_text(st))
    else:
        run(a.once)
