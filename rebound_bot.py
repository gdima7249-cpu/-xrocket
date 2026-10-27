"""Бот для Bybit (СПОТ, без плеча): правило «покупка после резкого падения, продажа через 1 час».

Правило B (проверено на 4 независимых выборках истории, см. rebound_research.py): цена монеты упала на 3% и больше за 15 минут
(три закрытые 5-минутные свечи) → маркет-покупка на фиксированную сумму USDT → маркет-продажа купленного ровно через 1 час.
Правило A (≥2% за одну 5-минутную свечу) включается флагом --rules A,B; по истории оно слабее.

Режимы (переменная BYBIT_ENV в .env или флаг --env):
  demo  — демо-счёт Bybit (виртуальные деньги, настоящие котировки). ПО УМОЛЧАНИЮ. Ключи создаются в демо-режиме Bybit.
  real  — реальные деньги. Включается только явно: --env real И флаг --i-understand-real-money.
Ключи (только в .env, никогда в чат): BYBIT_API_KEY, BYBIT_API_SECRET. Права ключа: только «Спот — торговля».
НИКОГДА не давайте ключу право на вывод средств. Лучше привязать ключ к IP сервера.

Комиссия спота Bybit — около 0.1% за сторону (0.2% на сделку, больше, чем в исследовании): перевес правила B по истории
+0.4…1% на сделку до этой поправки, после неё — тоньше; правило A на споте почти не окупается.

Ограничители: сумма на сделку (--stake, по умолчанию 5 USDT), не больше --max-open позиций одновременно (по умолчанию 2),
стоп-кран по суммарному убытку (--max-loss, USDT): после него новых покупок нет. Каждая сделка — в rebound_bot.csv с реальными
ценами исполнения и ценой сигнала (так видно проскальзывание). Состояние — rebound_bot.json: при перезапуске бот продолжит
и продаст то, что купил, вовремя.

  pip install ccxt
  python rebound_bot.py --selftest              # проверка логики без биржи
  python rebound_bot.py --check                 # подключиться (демо), показать баланс и минимальные суммы ордеров, ничего не покупать
  python rebound_bot.py                         # работать на демо-счёте
  python rebound_bot.py --top 40                # следить за 40 самыми торгуемыми монетами Bybit (больше сигналов)
  python rebound_bot.py --report
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone

try:
    import config  # noqa: F401  (подгружает .env)
except Exception:  # noqa: BLE001
    pass

COINS = "SOL,XRP,DOGE,ADA,AVAX,LINK,LTC,DOT,NEAR,ATOM,BTC,ETH"
BAR = 300
HOLD_SEC = 3600
RULES = {"A": (1, 0.02), "B": (3, 0.03)}
STATE, LOG = "rebound_bot.json", "rebound_bot.csv"


def make_exchange(env):
    import ccxt
    ex = ccxt.bybit({"apiKey": os.getenv("BYBIT_API_KEY", ""), "secret": os.getenv("BYBIT_API_SECRET", ""),
                     "enableRateLimit": True, "options": {"defaultType": "spot", "adjustForTimeDifference": True}})
    if env == "demo":
        ex.enable_demo_trading(True)
    return ex


STABLE = {"USDC", "USDE", "FDUSD", "DAI", "TUSD", "BUSD", "USDD", "PYUSD", "USD1", "EUR", "EURC", "EURT", "WBTC", "STETH",
          "WETH", "XAUT", "PAXG", "USTC"}


def top_coins(ex, n, min_volume, always=()):
    """n самых торгуемых монет к USDT на споте (по обороту за сутки), без стейблкоинов, обёрток и токенов с плечом."""
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or ":" in sym:
            continue
        base = sym.split("/")[0]
        if base in STABLE or base.endswith(("3L", "3S", "2L", "2S", "5L", "5S", "UP", "DOWN")):
            continue
        vol = float(t.get("quoteVolume") or 0)
        if vol >= min_volume:
            rows.append((vol, base))
    rows.sort(reverse=True)
    out = [b for _, b in rows[:n]]
    for b in always:                                                 # монеты, по которым уже открыта позиция, не теряем
        if b not in out:
            out.append(b)
    return out


class Bot:
    def __init__(self, ex, coins, rules, stake, max_open, max_loss, log=print, state_path=STATE, log_path=LOG):
        self.ex, self.coins, self.rules = ex, coins, rules
        self.stake, self.max_open, self.max_loss, self.log = stake, max_open, max_loss, log
        self.state_path, self.log_path = state_path, log_path
        self.st = self._load()

    # --- состояние ---
    def _load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        return {"last": {}, "open": [], "closed": []}

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.st, f, ensure_ascii=False)
        os.replace(tmp, self.state_path)

    def realized(self):
        return sum(c["pnl"] for c in self.st["closed"])

    # --- торговля ---
    def buy(self, coin, rule, signal_price, move, now=None):
        sym = f"{coin}/USDT"
        o = self.ex.create_market_buy_order_with_cost(sym, self.stake)
        o = self._settle(o, sym)
        qty, cost = float(o.get("filled") or 0), float(o.get("cost") or 0)
        fee_base = self._fee_in(o, coin)
        if qty <= 0:
            self.log(f"  {coin}: покупка не исполнилась ({o.get('status')})")
            return
        price = cost / qty
        pos = {"coin": coin, "rule": rule, "t": now or time.time(), "qty": qty - fee_base, "cost": cost, "price": price,
               "signal_price": signal_price, "move": move}
        self.st["open"].append(pos)
        self.log(f"  КУПЛЕНО {coin} [{rule}] на {cost:.2f} USDT по {price:.6g} (цена сигнала {signal_price:.6g}, "
                 f"проскальзывание {100 * (price / signal_price - 1):+.2f}%); продажа через 1 ч")

    def sell(self, pos):
        sym = f"{pos['coin']}/USDT"
        free = float((self.ex.fetch_balance().get(pos["coin"]) or {}).get("free") or 0)
        qty = min(pos["qty"], free)
        qty = float(self.ex.amount_to_precision(sym, qty))
        if qty <= 0:
            self.log(f"  {pos['coin']}: нечего продавать (баланс {free}) — позиция снята")
            return 0.0
        o = self._settle(self.ex.create_order(sym, "market", "sell", qty), sym)
        got = float(o.get("cost") or 0) - self._fee_in(o, "USDT")
        pnl = got - pos["cost"]
        rec = {"время_покупки": datetime.fromtimestamp(pos["t"], timezone.utc).strftime("%Y-%m-%d %H:%M"), "монета": pos["coin"],
               "правило": pos["rule"], "падение": round(pos["move"], 4), "цена_сигнала": pos["signal_price"],
               "цена_покупки": pos["price"], "цена_продажи": (float(o.get("cost") or 0) / qty) if qty else 0,
               "потрачено": round(pos["cost"], 4), "получено": round(got, 4), "pnl": round(pnl, 4)}
        self.st["closed"].append(rec)
        self._write(rec)
        self.log(f"  ПРОДАНО {pos['coin']} [{pos['rule']}]: {pos['cost']:.2f} → {got:.2f} USDT ({pnl:+.2f}, "
                 f"{100 * pnl / pos['cost']:+.2f}%)")
        return pnl

    def _settle(self, o, sym):
        """Ждёт, пока биржа сообщит итог маркет-ордера (цена и количество)."""
        for _ in range(10):
            if o.get("filled") and o.get("cost"):
                return o
            time.sleep(0.5)
            for getter in ("fetch_closed_order", "fetch_order"):
                try:
                    o = getattr(self.ex, getter)(o["id"], sym, {"acknowledged": True}) if getter == "fetch_order" \
                        else getattr(self.ex, getter)(o["id"], sym)
                    break
                except Exception:  # noqa: BLE001
                    continue
        return o

    @staticmethod
    def _fee_in(o, cur):
        fees = o.get("fees") or ([o["fee"]] if o.get("fee") else [])
        return sum(float(f.get("cost") or 0) for f in fees if f and f.get("currency") == cur)

    def _write(self, rec):
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(list(rec))
            w.writerow(list(rec.values()))

    # --- один проход ---
    def tick(self, now=None):
        now = now or time.time()
        # 1) продать то, что держим час
        for pos in list(self.st["open"]):
            if now - pos["t"] >= HOLD_SEC:
                try:
                    self.sell(pos)
                    self.st["open"].remove(pos)
                except Exception as e:  # noqa: BLE001
                    self.log(f"  {pos['coin']}: ошибка продажи ({type(e).__name__}: {str(e)[:100]}), повторю через минуту")
        # 2) новые сигналы по закрытым свечам
        halted = self.realized() <= -self.max_loss
        for coin in self.coins:
            try:
                bars = self.ex.fetch_ohlcv(f"{coin}/USDT", "5m", limit=10)
            except Exception as e:  # noqa: BLE001
                self.log(f"  {coin}: ошибка свечей ({type(e).__name__}: {str(e)[:80]})")
                continue
            closed = [b for b in bars if b[0] / 1000 + BAR <= now]
            if not closed:
                continue
            last_t = closed[-1][0]
            if self.st["last"].get(coin) == last_t:
                continue
            self.st["last"][coin] = last_t
            c = [b[4] for b in closed]
            for rule in self.rules:
                k, thr = RULES[rule]
                if len(c) <= k:
                    continue
                move = c[-1] / c[-1 - k] - 1
                if move > -thr:
                    continue
                if halted:
                    self.log(f"  СИГНАЛ {coin} [{rule}] {100 * move:.1f}%, но стоп-кран по убытку сработал — не покупаю")
                elif any(p["coin"] == coin for p in self.st["open"]):
                    pass
                elif len(self.st["open"]) >= self.max_open:
                    self.log(f"  СИГНАЛ {coin} [{rule}] {100 * move:.1f}%, но уже открыто {self.max_open} позиций — пропуск")
                else:
                    self.log(f"  СИГНАЛ {coin} [{rule}]: {100 * move:.2f}% за {k * 5} мин")
                    try:
                        self.buy(coin, rule, c[-1], move, now)
                    except Exception as e:  # noqa: BLE001
                        self.log(f"  {coin}: ошибка покупки ({type(e).__name__}: {str(e)[:120]})")
                break                                                # одна покупка на монету за свечу
        self.save()

    def report(self):
        cl = self.st["closed"]
        if not cl:
            return f"Закрытых сделок нет. Открыто: {len(self.st['open'])}."
        pnl = [c["pnl"] for c in cl]
        n = len(pnl)
        slip = [c["цена_покупки"] / c["цена_сигнала"] - 1 for c in cl if c.get("цена_сигнала")]
        return (f"Сделок {n}, в плюсе {100 * sum(p > 0 for p in pnl) / n:.0f}%, итог {sum(pnl):+.2f} USDT "
                f"(в среднем {sum(pnl) / n:+.3f} на сделку), среднее проскальзывание покупки {100 * sum(slip) / max(1, len(slip)):+.2f}%. "
                f"Открыто: {len(self.st['open'])}. Ожидание по истории: около +0.3…0.5% от суммы на сделку; для вывода нужно 100+ сделок.")


def selftest():
    import tempfile

    class FakeEx:
        def __init__(self):
            self.bal = {"USDT": 100.0}
            self.orders = []
            self.price = {}
            self.bars = {}

        def fetch_ohlcv(self, sym, tf, limit=10):
            return self.bars.get(sym, [])[-limit:]

        def create_market_buy_order_with_cost(self, sym, cost):
            p = self.price[sym] * 1.001                              # проскальзывание 0.1%
            q = cost / p
            coin = sym.split("/")[0]
            self.bal["USDT"] -= cost
            self.bal[coin] = self.bal.get(coin, 0) + q * 0.999       # комиссия 0.1% в монете
            o = {"id": str(len(self.orders)), "filled": q, "cost": cost, "status": "closed", "fees": [{"currency": coin, "cost": q * 0.001}]}
            self.orders.append(o)
            return o

        def create_order(self, sym, typ, side, qty):
            p = self.price[sym] * 0.999
            coin = sym.split("/")[0]
            self.bal[coin] -= qty
            got = qty * p
            self.bal["USDT"] += got * 0.999
            o = {"id": str(len(self.orders)), "filled": qty, "cost": got, "status": "closed", "fees": [{"currency": "USDT", "cost": got * 0.001}]}
            self.orders.append(o)
            return o

        def fetch_balance(self):
            return {k: {"free": v} for k, v in self.bal.items()}

        def amount_to_precision(self, sym, q):
            return f"{math.floor(q * 1e6) / 1e6:.6f}"

        def fetch_closed_order(self, oid, sym):
            return self.orders[int(oid)]

    d = tempfile.mkdtemp()
    ex = FakeEx()
    msgs = []
    bot = Bot(ex, ["SOL", "XRP"], ["B"], 5, 1, 3, log=msgs.append, state_path=os.path.join(d, "s.json"), log_path=os.path.join(d, "l.csv"))
    t0 = 1_700_000_000 // BAR * BAR
    # SOL: ровно 100, потом три свечи вниз до 96.5 (−3.5% за 15 мин); XRP: тоже падает, но позиция уже одна (max_open=1)
    sol = [100, 100, 100, 100, 99, 98, 96.5]
    xrp = [1, 1, 1, 1, 0.99, 0.98, 0.96]
    for name, series in (("SOL/USDT", sol), ("XRP/USDT", xrp)):
        ex.bars[name] = [[(t0 + i * BAR) * 1000, p, p, p, p, 0] for i, p in enumerate(series)]
    ex.price = {"SOL/USDT": 96.5, "XRP/USDT": 0.96}
    now = t0 + len(sol) * BAR + 1
    bot.tick(now)
    bought = len(bot.st["open"]) == 1 and bot.st["open"][0]["coin"] == "SOL"
    skipped = any("уже открыто 1" in m for m in msgs)
    bot.tick(now + 10)                                               # та же свеча — повторной покупки быть не должно
    no_dup = len(ex.orders) == 1
    ex.price["SOL/USDT"] = 98.0                                      # через час цена выше
    bot.tick(now + HOLD_SEC + 5)
    sold = not bot.st["open"] and len(bot.st["closed"]) == 1 and bot.st["closed"][0]["pnl"] > 0
    bot2 = Bot(ex, ["SOL"], ["B"], 5, 1, 3, log=msgs.append, state_path=os.path.join(d, "s.json"), log_path=os.path.join(d, "l.csv"))
    persisted = len(bot2.st["closed"]) == 1
    bot2.st["closed"].append({"pnl": -5, "цена_покупки": 1, "цена_сигнала": 1})
    ex.bars["SOL/USDT"].append([(t0 + 7 * BAR) * 1000, 90, 90, 90, 90, 0])
    ex.bars["SOL/USDT"].append([(t0 + 8 * BAR) * 1000, 88, 88, 88, 88, 0])
    ex.bars["SOL/USDT"].append([(t0 + 9 * BAR) * 1000, 85, 85, 85, 85, 0])
    bot2.tick(t0 + 10 * BAR + 1)
    halted = any("стоп-кран" in m for m in msgs) and not bot2.st["open"]
    class TEx:
        def fetch_tickers(self):
            return {"BTC/USDT": {"quoteVolume": 9e8}, "SOL/USDT": {"quoteVolume": 2e8}, "USDC/USDT": {"quoteVolume": 5e8},
                    "BTC3L/USDT": {"quoteVolume": 1e8}, "PEPE/USDT": {"quoteVolume": 1e6}, "ETH/USDC": {"quoteVolume": 1e9},
                    "XRP/USDT": {"quoteVolume": 7e7}}
    picked = top_coins(TEx(), 2, 5e6, always=["DOGE"])
    top_ok = picked == ["BTC", "SOL", "DOGE"]
    checks = [("выбор самых торгуемых монет без стейблов, плеча и мелочи", top_ok), ("покупка по сигналу B (−3.5% за 15 мин)", bought), ("лимит одновременных позиций соблюдается", skipped),
              ("свеча обрабатывается один раз", no_dup), ("продажа через час, прибыль посчитана", sold),
              ("состояние сохраняется между запусками", persisted), ("стоп-кран по убытку блокирует покупки", halted),
              ("отчёт строится", "Сделок 2" in bot2.report())]
    ok = 0
    for name, good in checks:
        print(("  ok   " if good else "  FAIL ") + name)
        ok += bool(good)
    print(f"\nИтог: {ok}/{len(checks)} проверок пройдено")
    return ok == len(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=os.getenv("BYBIT_ENV", "demo"), choices=["demo", "real"])
    ap.add_argument("--i-understand-real-money", action="store_true")
    ap.add_argument("--coins", default=COINS)
    ap.add_argument("--top", type=int, default=0, help="вместо --coins следить за N самыми торгуемыми монетами Bybit (например 40)")
    ap.add_argument("--min-volume", type=float, default=5e6, help="с --top: минимальный оборот за сутки, USDT (по умолчанию 5 млн)")
    ap.add_argument("--rules", default="B")
    ap.add_argument("--stake", type=float, default=5.0, help="USDT на сделку")
    ap.add_argument("--max-open", type=int, default=2)
    ap.add_argument("--max-loss", type=float, default=3.0, help="стоп-кран: суммарный убыток в USDT, после которого покупки прекращаются")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.report:
        print(Bot(None, [], [], 0, 0, 0).report())
        return
    if a.env == "real" and not a.i_understand_real_money:
        raise SystemExit("Режим реальных денег включается только с флагом --i-understand-real-money. Сначала поработайте на демо.")
    if not os.getenv("BYBIT_API_KEY"):
        raise SystemExit("Нет BYBIT_API_KEY / BYBIT_API_SECRET в .env (ключ с правом только «Спот — торговля», без вывода).")
    ex = make_exchange(a.env)
    coins = a.coins.split(",")
    if a.top:
        coins = top_coins(ex, a.top, a.min_volume)
        print(f"Самые торгуемые монеты Bybit (оборот ≥ {a.min_volume / 1e6:.0f} млн USDT за сутки): {len(coins)}")
    if a.check:
        ex.load_markets()
        bal = ex.fetch_balance()
        print(f"Bybit {'ДЕМО' if a.env == 'demo' else 'РЕАЛЬНЫЙ СЧЁТ'}: USDT свободно {(bal.get('USDT') or {}).get('free')}")
        for c in coins:
            m = ex.markets.get(f"{c}/USDT")
            if not m:
                print(f"  {c}/USDT: нет такого рынка на споте")
                continue
            lim = m.get("limits", {})
            print(f"  {c}/USDT: мин. сумма ордера {lim.get('cost', {}).get('min')} USDT, мин. количество {lim.get('amount', {}).get('min')}")
        print(f"Ставка {a.stake} USDT должна быть не меньше минимальной суммы ордера.")
        return
    bot = Bot(ex, coins, a.rules.split(","), a.stake, a.max_open, a.max_loss)
    print(f"Bybit {'ДЕМО' if a.env == 'demo' else 'РЕАЛЬНЫЕ ДЕНЬГИ'}: правила {a.rules}, ставка {a.stake} USDT, до {a.max_open} позиций, "
          f"стоп-кран −{a.max_loss} USDT. Монеты: {', '.join(coins)}. Ctrl+C — стоп (открытые позиции продадутся при следующем запуске).")
    print(bot.report())
    last_rep = last_top = time.time()
    while True:
        if a.top and time.time() - last_top > 6 * 3600:                # список самых торгуемых обновляется раз в 6 часов
            try:
                bot.coins = top_coins(ex, a.top, a.min_volume, always=[p["coin"] for p in bot.st["open"]])
            except Exception as e:  # noqa: BLE001
                print(f"  не удалось обновить список монет: {type(e).__name__}")
            last_top = time.time()
        try:
            bot.tick()
        except Exception as e:  # noqa: BLE001
            print(f"  ошибка прохода: {type(e).__name__}: {str(e)[:120]}")
        if time.time() - last_rep > 6 * 3600:
            print(datetime.now().strftime("[%Y-%m-%d %H:%M] ") + bot.report())
            last_rep = time.time()
        time.sleep(30)


if __name__ == "__main__":
    main()
