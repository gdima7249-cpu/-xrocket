"""Радар возможностей: знает несколько стратегий, раз в минуту смотрит на ВСЕ пары xRocket и показывает,
что сигналит прямо сейчас. Ничего не покупает (только публичные данные, ключ не нужен).

    python radar.py --once     # один проход: что видно сейчас, и выход
    python radar.py            # постоянная работа: виртуальный журнал сделок + отчёт раз в час

Виртуальный журнал: каждый сигнал входа открывает ВИРТУАЛЬНУЮ сделку (с комиссией и проскальзыванием из .env),
закрывается по тейку/стопу/таймауту/сигналу. За недели накапливается честная статистика по каждой стратегии «вперёд».
Арбитраж (USDT→A→B→USDT) считается по последним ценам тикера — это оптимистичная оценка (без стакана).
"""
import argparse
import json
import os
import time
from datetime import datetime

import ccxt
import pandas as pd

import config
from config import Params
from data import COLUMNS
from exchanges import make_exchange
from strategies import bollinger, dip, donchian, ema_rsi, trend_pullback

STATE_FILE = "radar_state.json"
MAX_PAIRS = int(os.getenv("RADAR_MAX_PAIRS", "40"))

# имя: таймфрейм, функция (df -> entries, exits), тейк, стоп, таймаут (часов)
STRATS = {
    "dip":            dict(tf="1h", fn=lambda d: (dip(d, config.SCAN_LOOKBACK, config.SCAN_DROP, config.SCAN_CONFIRM),
                                                  pd.Series(False, index=d.index)), tp=0.05, sl=0.05, hold=48),
    "donchian":       dict(tf="4h", fn=lambda d: donchian(d, config.DONCHIAN_IN, config.DONCHIAN_OUT),
                           tp=0.06, sl=0.03, hold=24 * 14),
    "ema_rsi":        dict(tf="1h", fn=lambda d: ema_rsi(d, 20, 50), tp=0.06, sl=0.03, hold=24 * 7),
    "trend_pullback": dict(tf="1h", fn=lambda d: trend_pullback(d, 200, 30), tp=0.06, sl=0.03, hold=24 * 7),
    "bollinger":      dict(tf="1h", fn=lambda d: bollinger(d, 20, 2.0), tp=0.05, sl=0.05, hold=48),
}


class Radar:
    def __init__(self):
        self.p = Params()
        self.ex = make_exchange("xrocket", "paper")
        self.universe = [s["symbol"] for s in self.ex.list_symbols()
                         if s["symbol"].endswith("-USDT") and s.get("enableTrading", True)][:MAX_PAIRS]
        self.s = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {
            "open": {}, "closed": [], "arb": {"checks": 0, "positive": 0, "best": None}}

    def save(self):
        json.dump(self.s, open(STATE_FILE, "w"))

    # ---------- виртуальные сделки ----------
    def open_trade(self, st, sym, price):
        self.s["open"][f"{st}|{sym}"] = {"entry": price * (1 + self.p.slippage), "t": time.time()}

    def close_trade(self, key, price, reason):
        st, sym = key.split("|")
        v = self.s["open"].pop(key)
        ret = price * (1 - self.p.slippage) * (1 - self.p.fee) / (v["entry"] * (1 + self.p.fee)) - 1
        self.s["closed"].append({"s": st, "sym": sym, "ret": ret, "reason": reason, "t": time.time()})
        return ret

    def manage(self, prices):
        """Выходы по тейку / стопу / таймауту (по последней цене)."""
        for key in list(self.s["open"]):
            st, sym = key.split("|")
            if sym not in prices:
                continue
            v, cfg = self.s["open"][key], STRATS[st]
            r = prices[sym] / v["entry"] - 1
            if r >= cfg["tp"]:
                self.close_trade(key, prices[sym], "take-profit")
            elif r <= -cfg["sl"]:
                self.close_trade(key, prices[sym], "stop-loss")
            elif (time.time() - v["t"]) / 3600 >= cfg["hold"]:
                self.close_trade(key, prices[sym], "timeout")

    # ---------- сканирование сигналов ----------
    def df(self, sym, tf):
        rows = self.ex.fetch_ohlcv(sym, tf, limit=260)[:-1]          # без незакрытой свечи
        return pd.DataFrame(rows, columns=COLUMNS)

    def scan(self, prices, tfs):
        """Сигналы входа и выхода на закрытых свечах. Возвращает список найденных сигналов входа."""
        found = []
        for sym in self.universe:
            if sym not in prices:
                continue
            frames = {}
            for st, cfg in STRATS.items():
                if cfg["tf"] not in tfs:
                    continue
                try:
                    if cfg["tf"] not in frames:
                        frames[cfg["tf"]] = self.df(sym, cfg["tf"])
                    d = frames[cfg["tf"]]
                except ccxt.ExchangeError:
                    break
                if len(d) < 60:
                    continue
                en, ex_ = cfg["fn"](d)
                key = f"{st}|{sym}"
                if key in self.s["open"] and bool(ex_.iloc[-1]):
                    self.close_trade(key, prices[sym], "signal")
                elif key not in self.s["open"] and bool(en.iloc[-1]):
                    self.open_trade(st, sym, prices[sym])
                    found.append((st, sym, prices[sym]))
        return found

    # ---------- арбитраж ----------
    def arbitrage(self, prices):
        """Лучший цикл USDT -> A -> B -> USDT по последним ценам: (брутто %, нетто %, описание)."""
        best = None
        fee = (1 - self.p.fee) ** 3
        bases = [s[:-5] for s in prices if s.endswith("-USDT")]
        for a in bases:
            for b in bases:
                if a == b:
                    continue
                if f"{a}-{b}" in prices:            # 1 A = p B  -> продаём A за B
                    rate, via = prices[f"{a}-{b}"], f"{a}-{b}"
                elif f"{b}-{a}" in prices:          # 1 B = p A  -> покупаем B за A
                    rate, via = 1 / prices[f"{b}-{a}"], f"{b}-{a}"
                else:
                    continue
                gross = (1 / prices[f"{a}-USDT"]) * rate * prices[f"{b}-USDT"]
                cand = ((gross - 1) * 100, (gross * fee - 1) * 100, f"USDT→{a}→{b}→USDT (через {via})")
                if best is None or cand[0] > best[0]:
                    best = cand
        if best:
            self.s["arb"]["checks"] += 1
            self.s["arb"]["positive"] += best[1] > 0
            self.s["arb"]["best"] = best
        return best

    # ---------- отчёт ----------
    def report(self, found, arb):
        print(f"\n{'=' * 78}\nРАДАР {datetime.now():%Y-%m-%d %H:%M}   (пар: {len(self.universe)}; "
              f"комиссия {self.p.fee*100:.2f}%/сторона, проскальзывание {self.p.slippage*100:.2f}%)")
        print("Сигналы входа сейчас:" if found else "Сигналов входа сейчас нет.")
        for st, sym, price in found:
            print(f"  {st:<15}{sym:<14}цена {price:.6g}  (тейк {STRATS[st]['tp']*100:.0f}% / стоп {STRATS[st]['sl']*100:.0f}%)")
        if arb:
            verdict = "ВОЗМОЖНОСТЬ (оптимистичная оценка!)" if arb[1] > 0 else "нет возможности"
            print(f"Арбитраж: лучший цикл {arb[2]}: брутто {arb[0]:+.2f}%, после комиссий {arb[1]:+.2f}% -> {verdict}")
            a = self.s["arb"]
            print(f"  за всё время: проверок {a['checks']}, с плюсом после комиссий: {a['positive']}")
        else:
            print("Арбитраж: нет подходящих связок пар.")
        print(f"Журнал виртуальных сделок (открыто {len(self.s['open'])}):")
        print(f"  {'стратегия':<15}{'сделок':>7}{'win%':>6}{'ср.% на сделку':>16}{'итого %':>9}")
        for st in STRATS:
            r = [c["ret"] * 100 for c in self.s["closed"] if c["s"] == st]
            if r:
                note = "  (мало данных, <30)" if len(r) < 30 else ""
                print(f"  {st:<15}{len(r):>7}{sum(x > 0 for x in r) / len(r) * 100:>6.0f}"
                      f"{sum(r) / len(r):>16.2f}{sum(r):>9.1f}{note}")
            else:
                print(f"  {st:<15}{0:>7}{'-':>6}{'-':>16}{'-':>9}")
        print("Не торговать, пока по стратегии нет сотен сделок и стабильного плюса после издержек.")


def run(once=False):
    r = Radar()
    last = {"1h": None, "4h": None}
    while True:
        try:
            prices = r.ex.fetch_tickers()
            r.manage(prices)
            now = time.time()
            tfs = [tf for tf, sec in (("1h", 3600), ("4h", 14400)) if last[tf] != int(now // sec)]
            found = []
            if tfs:
                time.sleep(0 if once else 5)
                for tf, sec in (("1h", 3600), ("4h", 14400)):
                    if tf in tfs:
                        last[tf] = int(now // sec)
                found = r.scan(prices, tfs)
                r.report(found, r.arbitrage(prices))
            else:
                r.arbitrage(prices)
            r.save()
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            print("Сеть/биржа:", e)
        if once:
            return
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    run(ap.parse_args().once)
