"""ДЕМО-бот Deriv: ставки ВВЕРХ (Rise/Fall, CALL) на 15 минут по реальным валютным парам после резкого падения.

Правило (проверено на истории 5 пар, deriv_forex_test.py): за последние 15 минут (три закрытые 5-минутные свечи) пара упала сильнее,
чем на 3 «обычных разброса» (разброс считается по последней неделе свечей) → ставка ВВЕРХ на 15 минут.
По истории такие ставки выигрывали около 54–57%. Порог безубыточности зависит от выплаты: 88% → 53.2%, 85% → 54.1%, 79% → 56%.
Поэтому бот ставит ТОЛЬКО на пары, у которых выплата на 15 минут прямо сейчас не ниже --min-payout (по умолчанию 85%).

Чтобы проверка шла быстрее, бот сам находит все валютные пары Deriv с нужной выплатой (раз в 2 часа перепроверяет) и следит за всеми.

ТОЛЬКО ДЕМО: подключение через deriv_demo_bot.connect() (номер демо-счёта, адрес /ws/demo), режима реальных денег нет.
Нужен .env: DERIV_TOKEN, DERIV_APP_ID, DERIV_ACCOUNT_ID (как для deriv_tick_bot.py).

  python deriv_fx_bot.py --scan                 # только показать пары и их выплаты на 15 минут (ничего не покупает)
  python deriv_fx_bot.py                        # работать (ставка 1 USD, лимит потерь 20 USD)
  python deriv_fx_bot.py --stake 1 --max-loss 30
  python deriv_fx_bot.py --selftest
Журнал ставок — deriv_log.csv (общий с другими Deriv-ботами), отчёт с процентом выигрышей и порогом печатается каждые 25 ставок.
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import deriv_demo_bot as B

PUBLIC_WS = os.getenv("DERIV_WS_URL", "wss://api.derivws.com/trading/v1/options/ws/public")
GRAN = 300
FALLBACK = ("frxEURUSD,frxGBPUSD,frxUSDJPY,frxAUDUSD,frxUSDCAD,frxUSDCHF,frxGBPCAD,frxEURGBP,frxEURJPY,frxGBPJPY,frxAUDJPY,"
            "frxEURCAD,frxEURAUD,frxEURCHF,frxGBPAUD,frxGBPCHF,frxAUDCAD,frxAUDNZD,frxNZDUSD,frxUSDNOK,frxUSDSEK").split(",")


class Public:
    """Публичная лента Deriv (свечи, список символов) с переподключением и ожиданием при лимите запросов."""

    def __init__(self, ws_factory=None):
        self.factory = ws_factory
        self.ws = None

    def call(self, req):
        import websocket
        for attempt in range(6):
            try:
                if self.ws is None:
                    self.ws = self.factory() if self.factory else websocket.create_connection(PUBLIC_WS, timeout=30)
                self.ws.send(json.dumps(req))
                msg = json.loads(self.ws.recv())
            except Exception:  # noqa: BLE001
                self.ws = None
                time.sleep(2 + 2 * attempt)
                continue
            err = msg.get("error")
            if err and "rate limit" in str(err.get("message", "")).lower():
                time.sleep(20 * (attempt + 1))
                continue
            if err:
                raise RuntimeError(f"Deriv API: {err.get('message')}")
            return msg
        raise RuntimeError("публичная лента Deriv не отвечает")

    def candles(self, symbol, count):
        """Закрытые 5-минутные свечи: список (время открытия, close) по возрастанию; незакрытая последняя свеча отбрасывается."""
        msg = self.call({"ticks_history": symbol, "style": "candles", "granularity": GRAN, "count": count, "end": "latest"})
        cs = [(int(c["epoch"]), float(c["close"])) for c in msg.get("candles") or []]
        now = time.time()
        return [c for c in cs if c[0] + GRAN <= now]

    def forex_symbols(self):
        try:
            msg = self.call({"active_symbols": "brief"})
            out = []
            for s in msg.get("active_symbols") or []:
                sym = s.get("underlying_symbol") or s.get("symbol") or ""
                if sym.startswith("frx"):
                    out.append(sym)
            return out or list(FALLBACK)
        except Exception:  # noqa: BLE001
            return list(FALLBACK)


def payout_15m(client, currency, symbol):
    pr = client.call({"proposal": 1, "amount": 10, "basis": "stake", "contract_type": "CALL", "currency": currency,
                      "duration": 15, "duration_unit": "m", "underlying_symbol": symbol})["proposal"]
    return float(pr["payout"]) / float(pr["ask_price"]) - 1


def log_returns(closes):
    return [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]


def is_signal(closes, vol, k=3.0):
    """Падение за последние 3 закрытые свечи сильнее k разбросов (разброс 15 минут = vol·√3)."""
    if len(closes) < 4 or vol <= 0:
        return False, 0.0
    move = math.log(closes[-1] / closes[-4])
    z = move / (vol * math.sqrt(3))
    return z <= -k, z


class FxBot:
    def __init__(self, pub, trader, min_payout, log=print):
        self.pub, self.tr, self.min_payout, self.log = pub, trader, min_payout, log
        self.vol, self.last, self.pairs, self.pays = {}, {}, [], {}

    def refresh_pairs(self, symbols):
        self.pairs, self.pays = [], {}
        for s in symbols:
            try:
                p = payout_15m(self.tr.c, self.tr.currency, s)
            except Exception:  # noqa: BLE001
                continue
            self.pays[s] = p
            if p >= self.min_payout:
                self.pairs.append(s)
        return self.pays

    def refresh_vol(self, symbol):
        cs = self.pub.candles(symbol, 2000)                         # около недели торговли
        r = log_returns([c for _, c in cs])
        if len(r) > 100:
            m = sum(r) / len(r)
            self.vol[symbol] = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))

    def step(self):
        """Проверить закрытые свечи по всем парам; при сигнале — поставить ВВЕРХ на 15 минут (если выплата ещё ≥ порога)."""
        busy = {o.get("base") for o in self.tr.open.values()}
        closed_open = (int(time.time()) // GRAN - 1) * GRAN           # время открытия последней закрытой свечи
        for s in self.pairs:
            if self.last.get(s) == closed_open:                     # эту свечу уже проверили — не тратим запрос
                continue
            if s not in self.vol:
                self.refresh_vol(s)
                if s not in self.vol:
                    continue
            cs = self.pub.candles(s, 6)
            if len(cs) < 4 or self.last.get(s) == cs[-1][0]:
                continue
            self.last[s] = cs[-1][0]
            sig, z = is_signal([c for _, c in cs], self.vol[s])
            if not sig:
                continue
            if s in busy:
                self.log(f"  СИГНАЛ {s} ({z:.1f} разброса), но по паре уже есть открытая ставка — пропуск")
                continue
            try:
                p = payout_15m(self.tr.c, self.tr.currency, s)
            except Exception as e:  # noqa: BLE001
                self.log(f"  СИГНАЛ {s}, но цену ставки получить не удалось: {str(e)[:80]}")
                continue
            if p < self.min_payout:
                self.log(f"  СИГНАЛ {s} ({z:.1f}), но выплата сейчас {100 * p:.0f}% < {100 * self.min_payout:.0f}% — пропуск")
                continue
            self.log(f"  СИГНАЛ {s}: падение {z:.1f} разброса за 15 мин, выплата {100 * p:.0f}% → ВВЕРХ на 15 мин")
            try:
                self.tr.place(s, 1, cs[-1][1], symbol=s)
            except Exception as e:  # noqa: BLE001
                self.log(f"  {s}: ставка не прошла: {str(e)[:120]}")


def selftest():
    import tempfile
    B.LOG = os.path.join(tempfile.mkdtemp(), "d.csv")
    ok = []
    # 1) расчёт сигнала: ровная цена с маленьким шумом и резкое падение в конце
    closes = [1.0 + 0.0001 * ((i * 7) % 5 - 2) for i in range(300)]
    vol = math.sqrt(sum(x * x for x in log_returns(closes)) / len(log_returns(closes)))
    calm, _ = is_signal(closes, vol)
    crash = closes + [closes[-1] * (1 - 8 * vol)] * 0 + [closes[-1] * (1 - 3 * vol), closes[-1] * (1 - 6 * vol), closes[-1] * (1 - 9 * vol)]
    sig, z = is_signal(crash, vol)
    ok.append(("спокойная цена — сигнала нет", not calm))
    ok.append((f"резкое падение — сигнал есть (z={z:.1f})", sig))

    # 2) полный проход на имитации: публичные свечи и демо-клиент
    now = int(time.time()) // GRAN * GRAN

    class FakePub:
        def __init__(self):
            self.data = {}

        def candles(self, s, count):
            return self.data[s][-count:]

    pub = FakePub()
    base = [(now - (400 - i) * GRAN, c) for i, c in enumerate(closes[:396])]
    pub.data["frxAAA"] = base + [(now - 4 * GRAN, closes[-1]), (now - 3 * GRAN, crash[-3]), (now - 2 * GRAN, crash[-2]),
                                 (now - GRAN, crash[-1])]
    pub.data["frxBBB"] = [(now - (400 - i) * GRAN, c) for i, c in enumerate(closes[:400])]
    fc = B.FakeClient(outcomes=["won"])
    tr = B.Trader(fc, 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    msgs = []
    bot = FxBot(pub, tr, 0.85, log=msgs.append)
    bot.refresh_pairs(["frxAAA", "frxBBB"])
    ok.append(("пары с выплатой ≥85% выбраны (имитация платит 92%)", bot.pairs == ["frxAAA", "frxBBB"]))
    bot.step()
    ok.append(("по упавшей паре сделана одна ставка ВВЕРХ", len(tr.open) == 1 and any("СИГНАЛ frxAAA" in m for m in msgs)))
    ok.append(("по спокойной паре ставки нет", not any("frxBBB" in m for m in msgs)))
    bot.step()
    ok.append(("та же свеча не даёт повторной ставки", fc.n == 1))
    tr.poll()
    ok.append(("итог ставки записан", len(tr.done) == 1 and tr.done[0]["res"] == 1))
    low = FxBot(pub, B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m"), 0.95, log=msgs.append)
    low.refresh_pairs(["frxAAA"])
    ok.append(("при выплате ниже порога пара не берётся", low.pairs == []))
    for name, good in ok:
        print(("  ok   " if good else "  FAIL ") + name)
    n = sum(g for _, g in ok)
    print(f"\nИтог: {n}/{len(ok)} проверок пройдено")
    return n == len(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--max-loss", type=float, default=20.0)
    ap.add_argument("--min-payout", type=float, default=0.85)
    ap.add_argument("--symbols", default="", help="свой список пар через запятую (по умолчанию — все валютные пары Deriv)")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)

    pub = Public()
    trader = B.Trader(B.connect(), a.stake, 15, 10 ** 9, a.max_loss, unit="m")
    bot = FxBot(pub, trader, a.min_payout)
    symbols = a.symbols.split(",") if a.symbols else pub.forex_symbols()
    pays = bot.refresh_pairs(symbols)
    print(f"Выплаты на 15 минут (CALL) по {len(pays)} парам:")
    for s, p in sorted(pays.items(), key=lambda x: -x[1]):
        print(f"  {s:<12}{100 * p:5.0f}%   порог {100 / (1 + p):.1f}%" + ("   ← берём" if s in bot.pairs else ""))
    if a.scan:
        return
    if not bot.pairs:
        raise SystemExit(f"Сейчас нет пар с выплатой ≥{100 * a.min_payout:.0f}% (рынок закрыт или выплаты снижены). Попробуйте позже.")
    print(f"\nРаботает на ДЕМО: {len(bot.pairs)} пар, ставка {a.stake}, лимит потерь {a.max_loss}. Ctrl+C — стоп.")
    last_pairs = last_rep = time.time()
    reported = 0
    while True:
        try:
            if time.time() - last_pairs > 2 * 3600:
                bot.refresh_pairs(symbols)
                bot.vol.clear()
                last_pairs = time.time()
                print(datetime.now().strftime("[%H:%M] ") + f"пары с выплатой ≥{100 * a.min_payout:.0f}%: {', '.join(bot.pairs) or 'нет'}")
            if trader.open:
                trader.poll()
            if trader.halted and not trader.open:
                print("Лимит потерь достигнут, бот остановлен.\nИТОГ: " + trader.report())
                return
            if not trader.halted:
                bot.step()
            getattr(trader.c, "idle_ping", lambda: None)()
            if len(trader.done) >= reported + 25 or time.time() - last_rep > 6 * 3600:
                print(datetime.now().strftime("[%H:%M] ") + trader.report())
                reported, last_rep = len(trader.done), time.time()
        except KeyboardInterrupt:
            print("\nИТОГ: " + trader.report())
            return
        except Exception as e:  # noqa: BLE001
            print(f"  ошибка прохода: {type(e).__name__}: {str(e)[:120]}")
        # спим до ближайшего закрытия 5-минутной свечи (+3 с), но не дольше 20 с — чтобы и входить сразу, и проверять ставки
        to_close = GRAN - (time.time() % GRAN) + 3
        time.sleep(min(20, to_close))


if __name__ == "__main__":
    main()
