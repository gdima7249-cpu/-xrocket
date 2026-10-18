"""Бот «вверх/вниз» для ДЕМО-счёта Deriv через официальное API (WebSocket): сигналы модели по реальным ценам Binance -> ставка на демо.

ТОЛЬКО ДЕМО. Бот сразу после подключения проверяет, что счёт виртуальный (is_virtual = 1); на реальном счёте он останавливается.
Режима реальных денег в нём нет сознательно. Токен создаётся в настройках Deriv (API token) на ДЕМО-счёте с правами Read и Trade;
хранится только в .env (DERIV_TOKEN=...), никому не присылайте.

Сигналы: та же модель (градиентный бустинг), что в signal_bot.py. Торгуем только крипту, которая есть на Deriv (BTC/USD, ETH/USD, ...),
по настоящим ценам; синтетические индексы (генерируются случайно) и OTC-подобные активы не используются.
Бот записывает КАЖДУЮ ставку и итог в deriv_log.csv и в отчёте сравнивает винрейт с порогом безубыточности по реальной выплате Deriv.

  pip install scikit-learn websocket-client
  python deriv_demo_bot.py --pairs BTC,ETH --stake 1 --expiry 5          # сигналы модели на крипте
  python deriv_demo_bot.py --mode random --symbol R_100 --unit t --expiry 5 --stake 1 --total 3000   # сам ставит 3000 раз (монетка)
  python deriv_demo_bot.py --mode random --symbol R_100 --unit t --expiry 5 --stake 2 --martingale 2 --max-stake 32 --total 1000
  python deriv_demo_bot.py --selftest        # проверка логики на имитации (без интернета и токена)

Лимиты безопасности: --max-trades-day (по умолчанию 40), --max-loss (потеря в валюте демо-счёта, после которой бот останавливается).
.env: DERIV_TOKEN=...  (необязательно DERIV_APP_ID=1089 — тестовый id по умолчанию; для своего приложения зарегистрируйте app_id на Deriv)
Не могу проверить это на реальном Deriv из своего окружения: если API вернёт ошибку, пришлите её текст (без токена).
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import config  # noqa: F401  (подгружает .env)
import signal_bot as S

TOKEN = os.getenv("DERIV_TOKEN", "")
APP_ID = os.getenv("DERIV_APP_ID", "1089")
WS_URL = os.getenv("DERIV_WS_URL", "wss://ws.derivws.com/websockets/v3")
LOG = "deriv_log.csv"
SYMBOLS = {"BTC": "cryBTCUSD", "ETH": "cryETHUSD", "LTC": "cryLTCUSD", "BCH": "cryBCHUSD"}   # крипта на Deriv (проверьте список у себя)


class DerivClient:
    """Минимальный клиент WebSocket API Deriv: запрос -> ответ с тем же req_id."""

    def __init__(self, app_id):
        import websocket
        url = f"{WS_URL}?app_id={app_id}" if "?" not in WS_URL else WS_URL
        try:
            self.ws = websocket.create_connection(url, timeout=30)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Не удалось подключиться к {url}: {type(e).__name__}: {str(e)[:150]}\n"
                             "Адрес API мог измениться или нужен свой app_id: смотрите документацию Deriv API (api.deriv.com), "
                             "задайте DERIV_WS_URL и DERIV_APP_ID в .env.")
        self.n = 0

    def call(self, req):
        self.n += 1
        req = dict(req, req_id=self.n)
        self.ws.send(json.dumps(req))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("req_id") == self.n:
                if msg.get("error"):
                    raise RuntimeError(f"Deriv API: {msg['error'].get('code')}: {msg['error'].get('message')}")
                return msg


class Trader:
    def __init__(self, client, stake, expiry, max_trades_day, max_loss, log=print, unit="m", martingale=1.0, max_stake=None):
        self.c, self.stake, self.expiry, self.log = client, stake, expiry, log
        self.base_stake, self.unit, self.martingale = stake, unit, martingale
        self.max_stake = max_stake or stake * 64
        self.max_trades_day, self.max_loss = max_trades_day, max_loss
        a = self.c.call({"authorize": TOKEN})["authorize"]
        if not a.get("is_virtual"):
            raise SystemExit("Это НЕ демо-счёт (is_virtual=0). Бот работает только на демо. Создайте токен на демо-счёте Deriv.")
        self.currency, self.balance0 = a.get("currency", "USD"), float(a.get("balance", 0))
        self.open, self.done, self.day_trades, self.day = {}, [], 0, datetime.now().date()
        self.halted = False
        self.log(f"Подключено к ДЕМО-счёту {a.get('loginid')}, баланс {self.balance0:.2f} {self.currency}.")

    def place(self, base, direction, entry_price, symbol=None):
        """Покупает контракт CALL/PUT на expiry (минут или тиков). Возвращает True, если ставка принята."""
        if self.halted:
            return False
        if datetime.now().date() != self.day:
            self.day, self.day_trades = datetime.now().date(), 0
        if self.day_trades >= self.max_trades_day:
            return False
        sym = symbol or SYMBOLS[base]
        prop = self.c.call({"proposal": 1, "amount": self.stake, "basis": "stake", "contract_type": "CALL" if direction > 0 else "PUT",
                            "currency": self.currency, "duration": self.expiry, "duration_unit": self.unit, "symbol": sym})["proposal"]
        buy = self.c.call({"buy": prop["id"], "price": float(prop["ask_price"])})["buy"]
        payout = float(prop.get("payout", 0))
        self.open[buy["contract_id"]] = {"base": base, "dir": direction, "t": time.time(), "entry": entry_price,
                                         "stake": float(buy["buy_price"]), "payout": payout}
        self.day_trades += 1
        self.log(f"  СТАВКА {base} {'ВВЕРХ' if direction > 0 else 'ВНИЗ'} на {self.expiry}{'т' if self.unit == 't' else ' мин'}, ставка {buy['buy_price']}, "
                 f"выплата {payout:.2f} (доля выплаты к ставке {payout / float(buy['buy_price']) - 1:.2f})")
        return True

    def poll(self):
        """Проверяет открытые контракты; закрытые записывает в журнал."""
        for cid in list(self.open):
            r = self.c.call({"proposal_open_contract": 1, "contract_id": cid})["proposal_open_contract"]
            if r.get("is_sold") or r.get("status") in ("won", "lost"):
                o = self.open.pop(cid)
                profit = float(r.get("profit", 0))
                o.update(profit=profit, res=1 if profit > 0 else (-1 if profit < 0 else 0))
                self.done.append(o)
                if self.martingale > 1:                                    # мартингейл с потолком ставки
                    self.stake = min(self.max_stake, self.stake * self.martingale) if o["res"] < 0 else self.base_stake
                self._write(o)
                self.log(f"  итог {o['base']}: {'ВЫИГРЫШ' if o['res'] > 0 else 'проигрыш' if o['res'] < 0 else 'ничья'} {profit:+.2f}")
                if sum(x["profit"] for x in self.done) <= -self.max_loss:
                    self.halted = True
                    self.log(f"ЛИМИТ ПОТЕРЬ достигнут ({self.max_loss}): новых ставок не будет.")

    def _write(self, o):
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["время", "монета", "направление", "ставка", "выплата", "прибыль", "итог"])
            w.writerow([datetime.fromtimestamp(o["t"]).strftime("%Y-%m-%d %H:%M:%S"), o["base"], "ВВЕРХ" if o["dir"] > 0 else "ВНИЗ",
                        o["stake"], o["payout"], o["profit"], o["res"]])

    def report(self):
        d = [x for x in self.done if x["res"] != 0]
        if not d:
            return "Пока нет закрытых ставок."
        n, w = len(d), sum(x["res"] > 0 for x in d)
        pay = sum(x["payout"] / x["stake"] - 1 for x in d) / n
        be = 1 / (1 + pay)
        profit = sum(x["profit"] for x in self.done)
        from demo_journal import wilson
        lo, hi = wilson(w, n)
        verdict = ("интервал целиком выше порога" if lo > be else "интервал целиком ниже порога" if hi < be
                   else "пока нельзя отличить от порога")
        return (f"Ставок {n}: угадано {100 * w / n:.1f}% (95%-интервал {100 * lo:.1f}…{100 * hi:.1f}%; порог при средней выплате "
                f"{100 * pay:.0f}%: {100 * be:.2f}%) — {verdict}; итог {profit:+.2f} {self.currency}"
                + ("  [мало ставок, <1000: выводов делать нельзя]" if n < 1000 else ""))


# ---------------- имитация для самопроверки ----------------
class FakeClient:
    def __init__(self, virtual=True, outcomes=None):
        self.virtual, self.outcomes, self.n, self.contracts = virtual, list(outcomes or []), 0, {}

    def call(self, req):
        if "authorize" in req:
            return {"authorize": {"is_virtual": 1 if self.virtual else 0, "loginid": "VRTC1", "balance": 10000, "currency": "USD"}}
        if "proposal" in req:
            return {"proposal": {"id": "p1", "ask_price": req["amount"], "payout": req["amount"] * 1.92}}
        if "buy" in req:
            self.n += 1
            self.contracts[self.n] = self.outcomes.pop(0) if self.outcomes else "won"
            return {"buy": {"contract_id": self.n, "buy_price": req["price"]}}
        if "proposal_open_contract" in req:
            res = self.contracts[req["contract_id"]]
            return {"proposal_open_contract": {"is_sold": 1, "status": res, "profit": 0.92 if res == "won" else -1.0}}
        raise AssertionError(req)


def selftest():
    import tempfile
    global LOG
    LOG = os.path.join(tempfile.mkdtemp(), "d.csv")
    ok = []
    try:
        Trader(FakeClient(virtual=False), 1, 5, 40, 20, log=lambda *a: None)
        ok.append(("реальный счёт отвергается", False))
    except SystemExit:
        ok.append(("реальный счёт отвергается", True))
    logs = []
    t = Trader(FakeClient(outcomes=["won", "lost", "lost"]), 1, 5, 40, 20, log=logs.append)
    for d in (1, -1, 1):
        t.place("BTC", d, 100.0)
    t.poll()
    ok.append(("3 ставки приняты, результаты записаны", len(t.done) == 3 and sum(x["res"] > 0 for x in t.done) == 1))
    ok.append(("прибыль считается: +0.92 -1 -1 = -1.08", abs(sum(x["profit"] for x in t.done) + 1.08) < 1e-9))
    ok.append(("отчёт показывает порог ≈52.08%", "52.08%" in t.report()))
    t2 = Trader(FakeClient(outcomes=["lost"] * 25), 1, 5, 40, 3, log=lambda *a: None)
    for _ in range(5):
        t2.place("ETH", 1, 100.0)
    t2.poll()
    ok.append(("лимит потерь останавливает бота", t2.halted and not t2.place("ETH", 1, 100.0)))
    t3 = Trader(FakeClient(), 1, 5, 2, 20, log=lambda *a: None)
    ok.append(("дневной лимит ставок", t3.place("BTC", 1, 1.0) and t3.place("BTC", 1, 1.0) and not t3.place("BTC", 1, 1.0)))
    ok.append(("журнал в CSV создан", os.path.exists(LOG)))
    tm = Trader(FakeClient(outcomes=["lost", "lost", "won", "lost"]), 2, 5, 10 ** 9, 1000, log=lambda *a: None, unit="t", martingale=2, max_stake=8)
    stakes = []
    for _ in range(4):
        stakes.append(tm.stake)
        tm.place("R_100", 1, 0.0, symbol="R_100")
        tm.poll()
    ok.append(("мартингейл: ставки 2,4,8, после выигрыша снова 2", stakes == [2, 4, 8, 2]))
    ta = Trader(FakeClient(outcomes=["won", "lost"] * 30), 1, 5, 10 ** 9, 1000, log=lambda *a: None, unit="t")
    run_auto(ta, "R_100", "random", 20, sleep=lambda s_: None, rnd=iter([0.1, 0.9] * 20).__next__, log=lambda *a: None)
    ok.append(("автономный режим сделал ровно 20 ставок", len(ta.done) == 20 and not ta.open))
    bad = 0
    for name, good in ok:
        print(("  ok   " if good else "  ОШИБКА  ") + name)
        bad += not good
    print(f"\nИтог: {len(ok) - bad}/{len(ok)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def run_auto(trader, symbol, mode, total, sleep=time.sleep, rnd=None, log=print, report_every=50):
    """Ставит сам: по одному контракту за раз (random — монетка, up/down — всегда в одну сторону), пока не наберётся total ставок."""
    import random
    rnd = rnd or random.random
    placed = 0
    while placed < total and not trader.halted:
        direction = (1 if rnd() < 0.5 else -1) if mode == "random" else (1 if mode == "up" else -1)
        if not trader.place(symbol, direction, 0.0, symbol=symbol):
            break
        placed += 1
        for _ in range(600):                                              # ждём закрытия контракта (до ~20 минут)
            trader.poll()
            if not trader.open:
                break
            sleep(2)
        if placed % report_every == 0:
            log(f"[{datetime.now():%H:%M}] {trader.report()}")
    log("ИТОГ: " + trader.report())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="BTC,ETH")
    ap.add_argument("--stake", type=float, default=1.0, help="ставка в валюте демо-счёта")
    ap.add_argument("--expiry", type=int, default=5, help="срок ставки в минутах")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--top", type=float, default=0.2)
    ap.add_argument("--max-trades-day", type=int, default=40)
    ap.add_argument("--max-loss", type=float, default=20.0)
    ap.add_argument("--mode", default="model", choices=["model", "random", "up", "down"],
                    help="model — сигналы модели (крипта); random/up/down — автономные ставки на любом символе (--symbol)")
    ap.add_argument("--symbol", default="R_100", help="символ Deriv для режимов random/up/down (R_100, 1HZ100V, frxEURUSD, cryBTCUSD ...)")
    ap.add_argument("--unit", default="m", choices=["m", "t"], help="единица срока: m — минуты, t — тики")
    ap.add_argument("--total", type=int, default=1000, help="сколько ставок сделать в режимах random/up/down")
    ap.add_argument("--martingale", type=float, default=1.0, help="множитель ставки после проигрыша (1 — фиксированная ставка)")
    ap.add_argument("--max-stake", type=float, default=None, help="потолок ставки при мартингейле (по умолчанию ставка×64)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if not TOKEN:
        raise SystemExit("Впишите DERIV_TOKEN в .env (токен ДЕМО-счёта с правами Read и Trade; никому его не присылайте).")
    if a.mode != "model":
        trader = Trader(DerivClient(APP_ID), a.stake, a.expiry, 10 ** 9, a.max_loss, unit=a.unit, martingale=a.martingale, max_stake=a.max_stake)
        print(f"Автономные ставки на ДЕМО: режим {a.mode}, символ {a.symbol}, срок {a.expiry}{a.unit}, ставка {a.stake}, всего {a.total}. Остановка: Ctrl+C.")
        run_auto(trader, a.symbol, a.mode, a.total)
        return
    pairs = [p for p in a.pairs.split(",") if p in SYMBOLS]
    if not pairs:
        raise SystemExit(f"Поддерживаемые монеты: {', '.join(SYMBOLS)}")
    trader = Trader(DerivClient(APP_ID), a.stake, a.expiry, a.max_trades_day, a.max_loss)
    model, thr = S.train(pairs, a.days, a.expiry, a.top)
    print("Работает на ДЕМО. Остановка: Ctrl+C.")
    last_min, last_report = 0, time.time()
    while not trader.halted or trader.open:
        now_min = int(time.time() // 60)
        if now_min != last_min:
            last_min = now_min
            trader.poll()
            if not trader.halted:
                for base in pairs:
                    if any(o["base"] == base for o in trader.open.values()):
                        continue
                    try:
                        sig = S.signal_for(model, thr, S.fetch_1m(base), a.expiry)
                        if sig:
                            trader.place(base, sig[0], sig[2])
                    except Exception as e:  # noqa: BLE001
                        print(f"  {base}: {type(e).__name__}: {str(e)[:100]}")
            if time.time() - last_report >= 600 and trader.done:
                last_report = time.time()
                print(f"[{datetime.now():%H:%M}] {trader.report()}")
        time.sleep(5)
    print(trader.report())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено.")
