"""Бот трендовых шортов (donchian_ls, ema_ls) для ТЕСТОВОГО фьючерсного счёта Binance (виртуальные деньги биржи).

Те же правила, что в replay_ls.py / forward_ls.py: сигналы на закрытых 4-часовых свечах, шорт USDT-M фьючерса с плечом 1x,
стоп 6% и тейк 20% ставятся на бирже сразу после входа (reduceOnly), выход по обратному сигналу или таймауту — рыночным ордером.
Размер позиции — доля капитала (по умолчанию 5%), не более 10 позиций, одна позиция на монету.

БЕЗОПАСНОСТЬ: бот работает ТОЛЬКО на тестовой сети (LS_ENV=testnet) или демо-счёте (LS_ENV=demo). Режима реальных денег в нём нет
сознательно. Ключи — только в .env (LS_API_KEY, LS_API_SECRET), никому не присылайте. Ключи тестовой сети отдельные от настоящих.

  python ls_bot.py --dry-run        # только показывает, что сделал бы (ордера не отправляются; ключи нужны для баланса)
  python ls_bot.py --once           # один проход и выход
  python ls_bot.py                  # постоянная работа (проверка раз в минуту, действия — на закрытии 4-часовых свечей)
  python ls_bot.py --stats          # сводка из файла состояния
  python ls_bot.py --selftest       # проверка логики на имитации биржи (без интернета и ключей)

Где взять ключи: тестовая сеть https://testnet.binancefuture.com (вход через GitHub/Google, создайте API Key) -> LS_ENV=testnet;
демо Binance (demo.binance.com, ключи в разделе Demo Trading) -> LS_ENV=demo.
.env: LS_ENV=testnet  LS_API_KEY=...  LS_API_SECRET=...  (необязательно: LS_POS_PCT=0.05 LS_MAX_POS=10 LS_MAX_LOSS=0.15 LS_STRATS=donchian_ls,ema_ls)
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config  # noqa: F401  (подгружает .env)
from replay import PAIRS
from replay_ls import STRATS

STATE_FILE = os.getenv("LS_STATE", "ls_bot_state.json")
ENV = os.getenv("LS_ENV", "testnet")
API_KEY, API_SECRET = os.getenv("LS_API_KEY", ""), os.getenv("LS_API_SECRET", "")
POS_PCT = float(os.getenv("LS_POS_PCT", "0.05"))
MAX_POS = int(os.getenv("LS_MAX_POS", "10"))
MAX_LOSS = float(os.getenv("LS_MAX_LOSS", "0.15"))               # стоп-кран: потеря 15% капитала -> закрыть всё и остановиться
NAMES = [s for s in os.getenv("LS_STRATS", "donchian_ls,ema_ls").split(",") if s]
PAIRS_USED = [s for s in os.getenv("LS_PAIRS", ",".join(PAIRS)).split(",") if s]
FEE = float(os.getenv("LS_FEE", "0.0005"))                       # тейкер-комиссия за сторону для оценки результата
BAR = 4 * 3600
KLINES = "https://data-api.binance.vision/api/v3/klines"


def fetch_candles(base: str, limit: int = 320) -> pd.DataFrame:
    """Закрытые 4-часовые свечи (цены реального рынка Binance — публичные данные, ключи не нужны)."""
    r = requests.get(KLINES, params={"symbol": base + "USDT", "interval": "4h", "limit": limit}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    return df[df["time"] + BAR * 1000 <= time.time() * 1000].reset_index(drop=True)


def make_exchange():
    import ccxt
    if ENV not in ("testnet", "demo"):
        raise SystemExit("LS_ENV должен быть testnet или demo. Режима реальных денег в этом боте нет сознательно.")
    if not API_KEY or not API_SECRET:
        raise SystemExit("Впишите LS_API_KEY и LS_API_SECRET в .env (ключи ТЕСТОВОЙ сети; не присылайте их никому).")
    ex = ccxt.binanceusdm({"enableRateLimit": True, "apiKey": API_KEY, "secret": API_SECRET})
    if ENV == "testnet":
        ex.set_sandbox_mode(True)
    else:
        ex.enable_demo_trading(True)
    return ex


class LSBot:
    def __init__(self, ex, candles_fn=fetch_candles, dry=False, state_file=STATE_FILE, log=print):
        self.ex, self.candles, self.dry, self.state_file, self.log = ex, candles_fn, dry, state_file, log
        self.ex.load_markets()
        self.s = json.load(open(state_file)) if os.path.exists(state_file) else {
            "open": {}, "closed": [], "bars": {}, "start_equity": None, "halted": False}
        if self.s["start_equity"] is None:
            self.s["start_equity"] = self.equity()
        self.symbols = {b: f"{b}/USDT:USDT" for b in PAIRS_USED if f"{b}/USDT:USDT" in self.ex.markets}
        self.log(f"Биржа: {ENV}; монет с фьючерсом: {len(self.symbols)} из {len(PAIRS_USED)}; капитал {self.equity():.2f} USDT; "
                 f"{'DRY-RUN (ордера не отправляются)' if dry else 'ордера отправляются на тестовый счёт'}")

    def save(self):
        json.dump(self.s, open(self.state_file, "w"), ensure_ascii=False, indent=1)

    def equity(self) -> float:
        return float(self.ex.fetch_balance()["total"].get("USDT", 0.0))

    # ---------------- сделки ----------------
    def open_short(self, base, strat, price):
        sym = self.symbols[base]
        _, tp_pct, sl_pct, _ = STRATS[strat]
        amount = float(self.ex.amount_to_precision(sym, self.equity() * POS_PCT / price))
        lim = self.ex.market(sym)["limits"]
        if amount <= 0 or amount < ((lim.get("amount") or {}).get("min") or 0) or amount * price < ((lim.get("cost") or {}).get("min") or 0):
            self.log(f"  {base}: сумма слишком мала для минимального ордера биржи — пропуск")
            return False
        if self.dry:
            self.log(f"  DRY: открыл бы шорт {base} ({strat}) на {amount} по ~{price:.6g}")
            return False
        try:
            self.ex.set_margin_mode("isolated", sym)
        except Exception:  # noqa: BLE001  (если уже установлен — биржа возвращает ошибку)
            pass
        try:
            self.ex.set_leverage(1, sym)
        except Exception:  # noqa: BLE001
            pass
        o = self.ex.create_order(sym, "market", "sell", amount)
        fill = float(o.get("average") or o.get("price") or price)
        sl_p, tp_p = fill * (1 + sl_pct), fill * (1 - tp_pct)
        ids = {}
        try:
            for tag, typ, p in (("sl", "STOP_MARKET", sl_p), ("tp", "TAKE_PROFIT_MARKET", tp_p)):
                od = self.ex.create_order(sym, typ, "buy", amount, None,
                                          {"stopPrice": float(self.ex.price_to_precision(sym, p)), "reduceOnly": True,
                                           "workingType": "MARK_PRICE"})
                ids[tag] = od["id"]
        except Exception as e:  # noqa: BLE001  — позиция без стопа недопустима: закрываем сразу
            self.log(f"  {base}: НЕ УДАЛОСЬ поставить стоп/тейк ({type(e).__name__}: {str(e)[:120]}) — закрываю позицию")
            self.ex.create_order(sym, "market", "buy", amount, None, {"reduceOnly": True})
            for oid in ids.values():
                try:
                    self.ex.cancel_order(oid, sym)
                except Exception:  # noqa: BLE001
                    pass
            return False
        self.s["open"][base] = {"strat": strat, "entry": fill, "amount": amount, "t": time.time(), **{k + "_id": v for k, v in ids.items()},
                                "sl": sl_p, "tp": tp_p}
        self.log(f"  ШОРТ {base} ({strat}): {amount} по {fill:.6g}, стоп {sl_p:.6g}, тейк {tp_p:.6g}")
        return True

    def record_close(self, base, exit_price, reason):
        v = self.s["open"].pop(base)
        ret = (v["entry"] - exit_price) / v["entry"] - 2 * FEE
        self.s["closed"].append({"base": base, "s": v["strat"], "entry": v["entry"], "exit": exit_price, "ret": ret,
                                 "reason": reason, "t0": v["t"], "t": time.time()})
        self.log(f"  закрыт {base} ({v['strat']}): {reason}, вход {v['entry']:.6g} -> выход {exit_price:.6g}, {100 * ret:+.2f}%")

    def close_short(self, base, reason, price):
        sym, v = self.symbols[base], self.s["open"][base]
        if self.dry:
            self.log(f"  DRY: закрыл бы {base} ({reason})")
            return
        for k in ("sl_id", "tp_id"):
            try:
                self.ex.cancel_order(v[k], sym)
            except Exception:  # noqa: BLE001
                pass
        o = self.ex.create_order(sym, "market", "buy", v["amount"], None, {"reduceOnly": True})
        self.record_close(base, float(o.get("average") or o.get("price") or price), reason)

    def sync(self):
        """Позиции, которые биржа закрыла сама (сработал стоп или тейк): записываем результат, снимаем второй ордер."""
        for base in list(self.s["open"]):
            sym, v = self.symbols[base], self.s["open"][base]
            pos = [p for p in self.ex.fetch_positions([sym]) if abs(float(p.get("contracts") or 0)) > 0]
            if pos:
                continue
            price, reason = None, "закрыта биржей"
            for k, name in (("sl_id", "стоп-лосс"), ("tp_id", "тейк-профит")):
                try:
                    od = self.ex.fetch_order(v[k], sym)
                    if od.get("status") == "closed":
                        price, reason = float(od.get("average") or od.get("price") or v["entry"]), name
                except Exception:  # noqa: BLE001
                    pass
            for k in ("sl_id", "tp_id"):
                try:
                    self.ex.cancel_order(v[k], sym)
                except Exception:  # noqa: BLE001
                    pass
            self.record_close(base, price or v["entry"], reason)

    # ---------------- цикл ----------------
    def kill_switch(self):
        eq = self.equity()
        if eq <= self.s["start_equity"] * (1 - MAX_LOSS) and not self.s["halted"]:
            self.log(f"СТОП-КРАН: капитал {eq:.2f} <= {100 * (1 - MAX_LOSS):.0f}% от старта {self.s['start_equity']:.2f}. Закрываю всё.")
            for base in list(self.s["open"]):
                self.close_short(base, "стоп-кран", self.s["open"][base]["entry"])
            self.s["halted"] = True
        return self.s["halted"]

    def step(self):
        if not self.dry:
            self.sync()
        if self.kill_switch():
            self.save()
            return
        expected_bar = (int(time.time()) // BAR) * BAR - BAR              # время открытия последней ЗАКРЫТОЙ 4-часовой свечи
        for base in self.symbols:
            if self.s["bars"].get(base, 0) >= expected_bar:
                continue
            try:
                df = self.candles(base)
                if len(df) < 120:
                    continue
                bar = int(df["time"].iloc[-1] / 1000)
                if bar < expected_bar:                                   # свеча ещё не опубликована
                    continue
                self.s["bars"][base] = bar
                price = float(df["close"].iloc[-1])
                held = self.s["open"].get(base)
                for st in NAMES:
                    fn, tp, sl, hold = STRATS[st]
                    le, lx, se, sx = (x.to_numpy(bool)[-1] for x in fn(df))
                    if held and held["strat"] == st:
                        if sx:
                            self.close_short(base, "сигнал выхода", price)
                        elif time.time() - held["t"] >= hold * BAR:
                            self.close_short(base, "таймаут", price)
                    elif not held and se and len(self.s["open"]) < MAX_POS:
                        if self.open_short(base, st, price):
                            held = self.s["open"][base]
            except Exception as e:  # noqa: BLE001  — одна монета не должна останавливать бота
                self.log(f"  {base}: {type(e).__name__}: {str(e)[:120]}")
        self.save()

    def stats(self):
        c = self.s["closed"]
        lines = [f"Капитал {self.equity():.2f} USDT (старт {self.s['start_equity']:.2f}); открыто {len(self.s['open'])}, закрыто {len(c)}"
                 + ("; ОСТАНОВЛЕН СТОП-КРАНОМ" if self.s["halted"] else "")]
        for st in NAMES:
            r = [x["ret"] * 100 for x in c if x["s"] == st]
            if r:
                lines.append(f"  {st:<14}{len(r):>5} сделок, win {100 * np.mean([v > 0 for v in r]):.0f}%, ср. {np.mean(r):+.2f}%, итого {sum(r):+.1f}%")
        for b, v in self.s["open"].items():
            lines.append(f"  открыт: {b:<6}{v['strat']:<12}вход {v['entry']:.6g}, стоп {v['sl']:.6g}, тейк {v['tp']:.6g}, "
                         f"с {datetime.fromtimestamp(v['t']):%m-%d %H:%M}")
        return "\n".join(lines)


# ---------------- имитация биржи для самопроверки ----------------
class FakeEx:
    """Минимальная имитация ccxt-биржи: рынки, баланс, позиции, рыночные и условные ордера."""

    def __init__(self, symbols, equity=1000.0):
        self.markets = {s: {} for s in symbols}
        self.cash, self.pos, self.orders, self.price, self.n = equity, {}, {}, {}, 0

    def load_markets(self): pass
    def market(self, s): return {"limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}}}
    def amount_to_precision(self, s, a): return f"{a:.3f}"
    def price_to_precision(self, s, p): return f"{p:.4f}"
    def fetch_balance(self): return {"total": {"USDT": self.cash}}
    def set_margin_mode(self, m, s): pass
    def set_leverage(self, l, s): pass

    def fetch_positions(self, syms):
        return [{"symbol": s, "contracts": self.pos.get(s, 0.0)} for s in syms]

    def create_order(self, s, typ, side, amount, price=None, params=None):
        params, self.n = params or {}, self.n + 1
        oid = f"o{self.n}"
        if typ == "market":
            px = self.price[s]
            self.pos[s] = self.pos.get(s, 0.0) + (amount if side == "sell" else -amount)
            if side == "buy":
                pass
            self.orders[oid] = {"id": oid, "status": "closed", "average": px}
        else:
            self.orders[oid] = {"id": oid, "status": "open", "type": typ, "stop": params["stopPrice"], "sym": s, "amount": amount}
        return dict(self.orders[oid])

    def fetch_order(self, oid, s): return dict(self.orders[oid])

    def cancel_order(self, oid, s):
        if self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "canceled"

    def move(self, s, px):
        """Цена пошла: срабатывают стоп (выше) и тейк (ниже) для шорта."""
        self.price[s] = px
        for o in self.orders.values():
            if o["status"] == "open" and o["sym"] == s:
                if (o["type"] == "STOP_MARKET" and px >= o["stop"]) or (o["type"] == "TAKE_PROFIT_MARKET" and px <= o["stop"]):
                    o["status"], o["average"] = "closed", o["stop"]
                    self.pos[s] = 0.0


def selftest():
    import tempfile
    from replay_ls import make_synth
    logs, ok = [], []
    syms = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    ex = FakeEx(syms)
    ex.price = {"BTC/USDT:USDT": 100.0, "ETH/USDT:USDT": 50.0}
    state = os.path.join(tempfile.mkdtemp(), "st.json")
    global PAIRS_USED
    PAIRS_USED = ["BTC", "ETH"]
    bot = LSBot(ex, candles_fn=lambda b: None, state_file=state, log=logs.append)

    bot.open_short("BTC", "donchian_ls", 100.0)
    v = bot.s["open"]["BTC"]
    ok.append(("шорт открыт на 5% капитала (0.5 BTC по 100)", abs(v["amount"] - 0.5) < 1e-9 and ex.pos["BTC/USDT:USDT"] == 0.5))
    ok.append(("стоп 6% и тейк 20% выставлены на бирже", abs(v["sl"] - 106) < 1e-6 and abs(v["tp"] - 80) < 1e-6
               and ex.orders[v["sl_id"]]["status"] == "open" and ex.orders[v["tp_id"]]["status"] == "open"))
    ex.move("BTC/USDT:USDT", 79.0)                                   # цена упала -> сработал тейк
    bot.sync()
    c = bot.s["closed"][-1]
    ok.append(("тейк зафиксирован как take-profit, результат ≈ +20% минус комиссии", c["reason"] == "тейк-профит" and abs(c["ret"] - (0.20 - 0.001)) < 1e-6))
    ok.append(("после закрытия биржей второй ордер (стоп) отменён", ex.orders[v["sl_id"]]["status"] == "canceled"))

    bot.open_short("ETH", "ema_ls", 50.0)
    ex.move("ETH/USDT:USDT", 54.0)                                   # цена выросла на 8% -> стоп
    bot.sync()
    c = bot.s["closed"][-1]
    ok.append(("стоп зафиксирован как stop-loss, результат ≈ -6% минус комиссии", c["reason"] == "стоп-лосс" and abs(c["ret"] - (-0.06 - 0.001)) < 1e-6))
    ok.append(("позиций открытых нет, закрытых 2", len(bot.s["open"]) == 0 and len(bot.s["closed"]) == 2))

    ex.price["BTC/USDT:USDT"] = 100.0                                 # после прошлого теста цена была 79 — возвращаем
    bot.open_short("BTC", "donchian_ls", 100.0)                     # закрытие по сигналу выхода
    ex.price["BTC/USDT:USDT"] = 95.0
    bot.close_short("BTC", "сигнал выхода", 95.0)
    c = bot.s["closed"][-1]
    ok.append(("закрытие по сигналу: рыночный выкуп, шорт в плюс ≈ +5%", abs(c["ret"] - ((100 - 95) / 100 - 0.001)) < 1e-6 and ex.pos["BTC/USDT:USDT"] == 0.0))
    ok.append(("после ручного закрытия стоп и тейк сняты", all(o["status"] != "open" for o in ex.orders.values())))

    ex.cash = 800.0                                                   # потеря 20% капитала > лимита 15%
    bot.open_short("BTC", "donchian_ls", 100.0)
    halted = bot.kill_switch()
    ok.append(("стоп-кран при потере >15%: всё закрыто и бот остановлен", halted and len(bot.s["open"]) == 0))

    bad = 0
    for name, good in ok:
        print(("  ok   " if good else "  ОШИБКА  ") + name)
        bad += not good
    print(f"\nИтог: {len(ok) - bad}/{len(ok)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    bot = LSBot(make_exchange(), dry=a.dry_run)
    if a.stats:
        print(bot.stats())
        return
    last_beat = time.time()
    if not a.once:
        print("Работает. Действия — на закрытии 4-часовых свечей; строка «живой» раз в 10 минут. Остановка: Ctrl+C.")
    while True:
        bot.step()
        if a.once:
            print(bot.stats())
            return
        if time.time() - last_beat >= 600:
            last_beat = time.time()
            print(f"[{datetime.now():%H:%M}] живой: открыто {len(bot.s['open'])}, закрыто {len(bot.s['closed'])}, "
                  f"капитал {bot.equity():.2f}" + ("; ОСТАНОВЛЕН" if bot.s["halted"] else ""))
        time.sleep(60)


if __name__ == "__main__":
    main()
