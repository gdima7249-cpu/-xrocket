"""Виртуальный бот «вверх/вниз» (как бинарные опционы). ТОЛЬКО ВИРТУАЛЬНЫЕ ДЕНЬГИ, реальные ордера не отправляются.

Каждые BINARY_EXPIRY_SEC секунд (или BINARY_EXPIRY_MIN минут; по умолчанию 5 минут) по каждой паре xRocket несколько «предсказателей» делают виртуальную
ставку «вверх» или «вниз»; по истечении срока ставка закрывается: цена выше/ниже цены входа -> выигрыш +PAYOUT от ставки
или проигрыш -100% ставки (ничья — возврат). Цены берутся из публичного тикера xRocket (ключ не нужен).

    python binary_bot.py            # постоянная работа (отчёт раз в час, строка «работает» раз в минуту при коротких сроках)
    python binary_bot.py --stats    # статистика из файла

Безубыточность при выплате 85%: нужно угадывать >54.05% ставок. Предсказатель random — контроль (ожидаемо около 50% и минус).
"""
import argparse
import json
import math
import os
import random
import time
from collections import deque
from datetime import datetime

import ccxt

import config  # noqa: F401  (подгружает .env)
from exchanges import make_exchange

STATE_FILE = os.getenv("BINARY_STATE", "binary_state.json")
EXPIRY = int(float(os.getenv("BINARY_EXPIRY_SEC") or float(os.getenv("BINARY_EXPIRY_MIN", "5")) * 60))  # срок ставки, секунд
PAYOUT = float(os.getenv("BINARY_PAYOUT", "0.85"))              # выплата при выигрыше (доля от ставки)
START = float(os.getenv("BINARY_START", "1000"))                # виртуальный баланс каждого предсказателя
STAKE = START * float(os.getenv("BINARY_STAKE_PCT", "0.01"))    # фиксированная ставка (1% от старта)
EPS = float(os.getenv("BINARY_EPS", "0.0005"))                  # минимальное движение для сигнала (0.05%)
SAMPLE = int(os.getenv("BINARY_SAMPLE_SEC") or max(2, min(30, EXPIRY // 6)))   # опрос цен, секунд
BEAT = int(os.getenv("BINARY_BEAT_SEC") or (60 if EXPIRY <= 120 else 600))     # как часто печатать строку «работает»
WARM = max(EXPIRY * 3, 300)                                     # прогрев истории перед ставками
PREDICTORS = ["random", "always_up", "momentum", "reversal", "rsi"]
BREAKEVEN = 1 / (1 + PAYOUT)


class Binary:
    def __init__(self):
        self.s = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {
            "acc": {p: {"bal": START, "bets": 0, "wins": 0, "ties": 0} for p in PREDICTORS},
            "open": [], "since": time.time()}
        for p in PREDICTORS:                           # на случай добавления предсказателей
            self.s["acc"].setdefault(p, {"bal": START, "bets": 0, "wins": 0, "ties": 0})
        self.series, self.last_bucket, self.rng = {}, None, random.Random(7)

    def save(self):
        json.dump(self.s, open(STATE_FILE, "w"))

    # ---------- данные ----------
    def price_ago(self, sym, secs, now):
        for t, p in reversed(self.series.get(sym, [])):
            if t <= now - secs:
                return p
        return None

    def rsi(self, sym):
        pts = [p for _, p in list(self.series.get(sym, []))[-30:]]
        if len(pts) < 20:
            return None
        d = [b - a for a, b in zip(pts, pts[1:])]
        g, l = sum(x for x in d if x > 0), -sum(x for x in d if x < 0)
        return 100.0 if l == 0 and g > 0 else 50.0 if l == 0 else 100 - 100 / (1 + g / l)

    # ---------- предсказатели ----------
    def predict(self, name, sym, price, now):
        if name == "random":
            return self.rng.choice(["up", "down"])
        if name == "always_up":
            return "up"
        if name in ("momentum", "reversal"):
            ago = self.price_ago(sym, EXPIRY, now)
            if not ago:
                return None
            mv = price / ago - 1
            d = "up" if mv > EPS else "down" if mv < -EPS else None
            return d if name == "momentum" or d is None else ("down" if d == "up" else "up")
        if name == "rsi":
            r = self.rsi(sym)
            return None if r is None else "up" if r < 30 else "down" if r > 70 else None

    # ---------- ставки ----------
    def open_bets(self, prices, now):
        for sym, price in prices.items():
            dq = self.series.get(sym)
            if not dq or dq[0][0] > now - WARM:
                continue                                  # мало истории
            for name in PREDICTORS:
                d = self.predict(name, sym, price, now)
                if d:
                    self.s["open"].append({"p": name, "sym": sym, "dir": d, "entry": price, "t": now})

    def resolve(self, b, price):
        a = self.s["acc"][b["p"]]
        a["bets"] += 1
        if price is None or price == b["entry"]:
            a["ties"] += 1
            return
        win = (price > b["entry"]) == (b["dir"] == "up")
        a["wins"] += win
        a["bal"] += STAKE * PAYOUT if win else -STAKE

    def tick(self, prices, now):
        for sym, p in prices.items():
            dq = self.series.setdefault(sym, deque(maxlen=400))
            if not dq or now - dq[-1][0] >= SAMPLE * 0.9:
                dq.append((now, p))
        still = []
        for b in self.s["open"]:
            if now >= b["t"] + EXPIRY:
                self.resolve(b, prices.get(b["sym"]))
            else:
                still.append(b)
        self.s["open"] = still
        bucket = int(now // EXPIRY)
        if bucket != self.last_bucket:
            self.last_bucket = bucket
            self.open_bets(prices, now)


def fmt_expiry():
    return f"{EXPIRY} с" if EXPIRY < 120 else f"{EXPIRY // 60} мин"


def stats_text(s):
    days = (time.time() - s["since"]) / 86400
    lines = [f"Виртуальные ставки «вверх/вниз» (срок {fmt_expiry()}, выплата {PAYOUT * 100:.0f}%, "
             f"ставка {STAKE:.0f} из {START:.0f}); работает {days:.1f} дн.",
             f"Порог безубыточности: угадывать больше {BREAKEVEN * 100:.2f}% ставок.",
             f"  {'предсказатель':<14}{'ставок':>8}{'ничьих':>8}{'винрейт%':>10}{'прибыль':>10}{'на ставку%':>11}  вывод"]
    for p in PREDICTORS:
        a = s["acc"][p]
        n = a["bets"] - a["ties"]
        wr = a["wins"] / n if n else 0.0
        z = (wr - BREAKEVEN) / math.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n) if n > 0 else 0.0
        note = ("мало ставок" if n < 300 else "ЗНАЧИМО выше порога (z>3)" if z > 3
                else "ниже/около порога" if z < 2 else "возможно выше, нужно больше данных")
        pnl = a["bal"] - START
        roi = pnl / (n * STAKE) * 100 if n else 0.0          # средний результат на 100% ставки
        lines.append(f"  {p:<14}{a['bets']:>8}{a['ties']:>8}{wr * 100:>10.2f}{pnl:>+10.1f}{roi:>+11.1f}  {note}")
    lines.append("«На ставку%» — средний результат с каждой ставки (при 50% угадываний и выплате 85% это около -7.5%).")
    lines.append("Ничья (цена не изменилась) — возврат ставки. random — контроль. ОСТОРОЖНО: в растущем рынке «always_up» выглядит гением — это не умение, а рост рынка.")
    return "\n".join(lines)


def run():
    ex, bot = make_exchange("xrocket", "paper"), Binary()
    print("Бот «вверх/вниз» запущен (виртуальные деньги). История прогревается около "
          f"{WARM // 60} мин, затем ставки каждые {fmt_expiry()} (опрос цен раз в {SAMPLE} с). Остановка: Ctrl+C.")
    last_beat = last_report = time.time()
    while True:
        try:
            now = time.time()
            bot.tick(ex.fetch_tickers(), now)
            bot.save()
            if now - last_beat >= BEAT:
                last_beat = now
                print(f"[{datetime.now():%H:%M}] работает: открытых ставок {len(bot.s['open'])}, "
                      f"всего закрыто {sum(a['bets'] for a in bot.s['acc'].values())}")
            if now - last_report >= 3600:
                last_report = now
                print("\n" + "=" * 78 + f"\n{datetime.now():%Y-%m-%d %H:%M}\n" + stats_text(bot.s))
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            print("Сеть/биржа:", e)
        time.sleep(SAMPLE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    if ap.parse_args().stats:
        if not os.path.exists(STATE_FILE):
            raise SystemExit("Файла состояния ещё нет: сначала запустите бота.")
        print(stats_text(json.load(open(STATE_FILE))))
    else:
        run()
