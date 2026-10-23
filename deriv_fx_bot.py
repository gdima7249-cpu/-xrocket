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
import tg_notify

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


MINUTES = 15                                                         # срок ставки (меняется флагом --minutes)


def payout_15m(client, currency, symbol, stake=10.0):
    """Выплата в долях ставки для CALL на MINUTES минут — на ТОЙ сумме, которую бот реально ставит (на малых ставках она ниже)."""
    pr = client.call({"proposal": 1, "amount": stake, "basis": "stake", "contract_type": "CALL", "currency": currency,
                      "duration": MINUTES, "duration_unit": "m", "underlying_symbol": symbol})["proposal"]
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
    def __init__(self, pub, trader, min_payout, log=print, max_open=0, stake_pct=0.0, stake_min=0.5, stake_max=5.0):
        self.pub, self.tr, self.min_payout, self.log = pub, trader, min_payout, log
        self.max_open = max_open                                     # 0 — без ограничения
        self.stake_pct, self.stake_min, self.stake_max = stake_pct, stake_min, stake_max   # сложный процент (0 — выключен)
        self.cnt = {"сигналов": 0, "поставлено": 0, "пропуск: низкая выплата": 0, "пропуск: пара занята": 0,
                    "пропуск: лимит открытых": 0, "ошибка ставки": 0}                         # счётчики с запуска
        self.low_pays = []                                           # выплаты у пропущенных сигналов
        self.watch = None                                            # все пары под наблюдением (выплата проверяется в момент сигнала)
        self.k = 3.0                                                 # порог силы падения в «разбросах» (флаг --k)
        self.vol, self.last, self.pairs, self.pays = {}, {}, [], {}

    def refresh_pairs(self, symbols):
        self.pairs, self.pays, self.errors = [], {}, {}
        for s in symbols:
            try:
                p = payout_15m(self.tr.c, self.tr.currency, s, self.tr.stake)
            except Exception as e:  # noqa: BLE001
                self.errors[s] = str(e)[:160]
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
        for s in (self.watch or self.pairs):
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
            if any(b[0] - a[0] != GRAN for a, b in zip(cs[-4:], cs[-3:])):   # внутри окна перерыв в торговле (ночь, выходные)
                continue
            sig, z = is_signal([c for _, c in cs], self.vol[s], self.k)
            if not sig:
                continue
            self.cnt["сигналов"] += 1
            if s in busy:
                self.log(f"  СИГНАЛ {s} ({z:.1f} разброса), но по паре уже есть открытая ставка — пропуск")
                self.cnt["пропуск: пара занята"] += 1
                continue
            if self.max_open and len(self.tr.open) >= self.max_open:
                self.log(f"  СИГНАЛ {s} ({z:.1f}), но открыто уже {len(self.tr.open)} ставок (лимит {self.max_open}) — пропуск")
                self.cnt["пропуск: лимит открытых"] += 1
                continue
            try:
                p = payout_15m(self.tr.c, self.tr.currency, s, self.tr.stake)
            except Exception as e:  # noqa: BLE001
                self.log(f"  СИГНАЛ {s}, но цену ставки получить не удалось: {str(e)[:80]}")
                continue
            if p < self.min_payout:
                self.log(f"  СИГНАЛ {s} ({z:.1f}), но выплата сейчас {100 * p:.0f}% < {100 * self.min_payout:.0f}% — пропуск")
                self.cnt["пропуск: низкая выплата"] += 1
                self.low_pays.append(p)
                continue
            if self.stake_pct:                                      # ставка = доля текущего баланса, в пределах [мин, макс]
                try:
                    bal = float(self.tr.c.call({"balance": 1})["balance"]["balance"])
                    self.tr.stake = round(min(self.stake_max, max(self.stake_min, self.stake_pct * bal)), 2)
                except Exception:  # noqa: BLE001
                    self.tr.stake = self.stake_min
            self.log(f"  СИГНАЛ {s}: падение {z:.1f} разброса за 15 мин, выплата {100 * p:.0f}% → ВВЕРХ на {MINUTES} мин, "
                     f"ставка {self.tr.stake}")
            try:
                if self.tr.place(s, 1, cs[-1][1], symbol=s):
                    self.cnt["поставлено"] += 1
            except Exception as e:  # noqa: BLE001
                self.log(f"  {s}: ставка не прошла: {str(e)[:120]}")
                self.cnt["ошибка ставки"] += 1


def breakdown(path):
    """Разбивка журнала ставок: по парам, по выплате, по времени суток (UTC). Текст для Telegram и терминала."""
    import csv
    if not os.path.exists(path):
        return "Журнал пока пуст."
    with open(path, encoding="utf-8") as f:
        rows = [r for r in list(csv.reader(f))[1:] if len(r) > 6 and r[6] in ("1", "-1")]
    if not rows:
        return "Закрытых ставок в журнале пока нет."

    def line(name, rs):
        w = sum(r[6] == "1" for r in rs)
        n = len(rs)
        pay = sum(float(r[4]) / float(r[3]) - 1 for r in rs if float(r[3]) > 0) / n
        prof = sum(float(r[5]) for r in rs)
        return f"{name:<10} {n:>4} ст. {100 * w / n:5.1f}%  выпл.{100 * pay:3.0f}%  {prof:+.2f}"

    out = ["По парам:"]
    pairs = {}
    for r in rows:
        pairs.setdefault(r[1].replace("frx", ""), []).append(r)
    for k, rs in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        out.append("  " + line(k, rs))
    out.append("По выплате:")
    for name, lo, hi in (("<84%", -1, 0.84), ("84–86%", 0.84, 0.86), ("≥86%", 0.86, 9)):
        rs = [r for r in rows if lo <= float(r[4]) / float(r[3]) - 1 < hi]
        if rs:
            out.append("  " + line(name, rs))
    out.append("По времени (UTC, мск = +3):")
    for name, lo, hi in (("00–06", 0, 6), ("06–12", 6, 12), ("12–18", 12, 18), ("18–24", 18, 24)):
        rs = [r for r in rows if lo <= int(r[0][11:13] or 0) < hi]
        if rs:
            out.append("  " + line(name, rs))
    out.append("Пока в группе меньше ~100 ставок, её процент — во многом случайность.")
    return "\n".join(out)


def counters_text(bot):
    if bot is None:
        return ""
    c = bot.cnt
    extra = ""
    if bot.low_pays:
        extra = f" (выплата у пропущенных: от {100 * min(bot.low_pays):.0f}% до {100 * max(bot.low_pays):.0f}%)"
    return ("\n\nС запуска бота: сигналов " + str(c["сигналов"]) + ", поставлено " + str(c["поставлено"]) +
            "; пропущено: низкая выплата " + str(c["пропуск: низкая выплата"]) + extra +
            ", пара занята " + str(c["пропуск: пара занята"]) + ", лимит открытых " + str(c["пропуск: лимит открытых"]) +
            ", ошибки " + str(c["ошибка ставки"]) +
            f". Под наблюдением пар: {len(bot.watch or bot.pairs)}, из них сейчас с выплатой ≥ порога: {len(bot.pairs)}.")


def tg_handlers(trader, log_path=None, bot=None):
    """Ответы на команды из Telegram: баланс, открытые ставки, история (из журнала deriv_log.csv — переживает перезапуски)."""
    import csv

    def balance():
        bal = trader.c.call({"balance": 1})["balance"]
        made = sum(x["profit"] for x in trader.done)
        return (f"💰 Баланс ДЕМО-счёта: {float(bal.get('balance', 0)):.2f} {bal.get('currency', trader.currency)}\n"
                f"С запуска бота: {len(trader.done)} закрытых ставок, итог {made:+.2f}; открыто сейчас: {len(trader.open)}")

    def open_bets():
        if not trader.open:
            return "📈 Открытых ставок нет."
        now = time.time()
        rows = [f"{o['base'].replace('frx', '')}: ВВЕРХ, ставка {o['stake']}, выплата {o['payout']:.2f}, "
                f"до закрытия ~{max(0, int((o['t'] + trader.expiry * 60 - now) // 60))} мин" for o in trader.open.values()]
        return f"📈 Открытые ставки ({len(rows)}):\n" + "\n".join(rows)

    def history():
        path = log_path or B.LOG
        if not os.path.exists(path):
            return "📜 Журнал пока пуст."
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        if not rows:
            return "📜 Журнал пока пуст."
        res = [int(r[6]) for r in rows if len(r) > 6 and r[6].lstrip("-").isdigit()]
        prof = sum(float(r[5]) for r in rows if len(r) > 5)
        w, l = sum(x > 0 for x in res), sum(x < 0 for x in res)
        last = "\n".join(f"{r[0][5:16]} {r[1].replace('frx', '')} {'✅' if r[6] == '1' else '❌' if r[6] == '-1' else '➖'} {float(r[5]):+.2f}"
                         for r in rows[-10:])
        pct = f"{100 * w / (w + l):.1f}%" if w + l else "—"
        return (f"📜 Всего в журнале: {len(rows)} ставок, выиграно {w}, проиграно {l} ({pct}), итог {prof:+.2f}\n"
                f"Последние {min(10, len(rows))}:\n{last}\n\nОтчёт с запуска: {trader.report()}" + counters_text(bot))

    def stats():
        return "📊 Разбивка:\n" + breakdown(log_path or B.LOG)

    return {"balance": balance, "open": open_bets, "history": history, "stats": stats}


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
    gap_pub = FakePub()
    gap_pub.data["frxGAP"] = base[:-3] + [(now - 2 * GRAN - 48 * 3600, crash[-3]), (now - 2 * GRAN, crash[-2]),
                                          (now - GRAN, crash[-1])]
    gtr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    gbot = FxBot(gap_pub, gtr, 0.85, log=msgs.append)
    gbot.refresh_pairs(["frxGAP"])
    gbot.vol["frxGAP"] = vol
    gbot.step()
    ok.append(("падение через перерыв в торговле (выходные) не даёт ставки", not gtr.open))
    low = FxBot(pub, B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m"), 0.95, log=msgs.append)
    low.refresh_pairs(["frxAAA"])
    ok.append(("при выплате ниже порога пара не берётся", low.pairs == []))
    low.watch = ["frxAAA"]
    low.step()
    ok.append(("пара под наблюдением даже при низкой выплате: сигнал виден и учтён как пропуск",
               low.cnt["сигналов"] == 1 and low.cnt["пропуск: низкая выплата"] == 1 and not low.tr.open))
    # сложный процент: ставка = 2% баланса (имитация баланса 40) → 0.8; с потолком и минимумом
    class BalClient(B.FakeClient):
        def call(self, req):
            if "balance" in req:
                return {"balance": {"balance": 40.0, "currency": "USD"}}
            return super().call(req)

    ctr = B.Trader(BalClient(), 0.5, 30, 10 ** 9, 20, log=lambda *a: None, unit="m")
    cbot = FxBot(pub, ctr, 0.85, log=msgs.append, stake_pct=0.02, stake_min=0.5, stake_max=5)
    cbot.refresh_pairs(["frxAAA"])
    cbot.step()
    ok.append(("сложный процент: ставка 2% от баланса 40 = 0.8", len(ctr.open) == 1 and abs(ctr.stake - 0.8) < 1e-9))

    # команды Telegram: отвечаем только владельцу
    os.environ["TELEGRAM_CHAT_ID"] = "777"
    sent = []
    queue = [[{"update_id": 1, "message": {"chat": {"id": 1}, "text": "старое"}}],
             [{"update_id": 2, "message": {"chat": {"id": 777}, "text": "📈 Текущие ставки"}},
              {"update_id": 3, "message": {"chat": {"id": 999}, "text": "📜 История"}},
              {"update_id": 4, "message": {"chat": {"id": 777}, "text": "📜 История"}}]]
    cmds = tg_notify.Commands(tg_handlers(tr), get=lambda off: queue.pop(0) if queue else [])
    cmds.poll(reply=sent.append)
    ok.append(("Telegram: ответ только владельцу, на 2 команды", len(sent) == 2 and "Открытых ставок нет" in sent[0]
               and "Всего в журнале: 1" in sent[1]))
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
    ap.add_argument("--minutes", type=int, default=15, choices=[15, 30, 60], help="срок ставки в минутах")
    ap.add_argument("--k", type=float, default=3.0, help="порог силы падения в разбросах (по умолчанию 3; меньше — больше сигналов)")
    ap.add_argument("--only", default="", help="ставить только на эти пары (через запятую, например frxGBPNZD,frxEURNZD)")
    ap.add_argument("--stake-pct", type=float, default=0.0,
                    help="сложный процент: ставка = эта доля баланса (0.02 = 2%%), не меньше --stake и не больше --stake-max. 0 — выключено")
    ap.add_argument("--stake-max", type=float, default=5.0, help="потолок ставки при --stake-pct")
    ap.add_argument("--stats", action="store_true", help="показать разбивку журнала (по парам, выплате, времени) и выйти")
    ap.add_argument("--max-open", type=int, default=0, help="не больше стольких открытых ставок одновременно (0 — без лимита)."
                    " Сигналы приходят пачками по одной новости, и такие ставки выигрывают/проигрывают вместе")
    ap.add_argument("--symbols", default="", help="свой список пар через запятую (по умолчанию — все валютные пары Deriv)")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)

    global MINUTES
    MINUTES = a.minutes
    pub = Public()
    trader = B.Trader(B.connect(), a.stake, a.minutes, 10 ** 9, a.max_loss, unit="m")
    bot = FxBot(pub, trader, a.min_payout, max_open=a.max_open, stake_pct=a.stake_pct, stake_min=a.stake, stake_max=a.stake_max)
    symbols = a.symbols.split(",") if a.symbols else pub.forex_symbols()
    if a.only:
        symbols = [x for x in symbols if x in set(a.only.split(","))]
    pays = bot.refresh_pairs(symbols)
    print(f"Выплаты на {MINUTES} минут (CALL, ставка {a.stake}) по {len(pays)} парам:")
    for s, p in sorted(pays.items(), key=lambda x: -x[1]):
        print(f"  {s:<12}{100 * p:5.0f}%   порог {100 / (1 + p):.1f}%" + ("   ← берём" if s in bot.pairs else ""))
    if bot.errors:
        print(f"\nНе удалось получить выплату по {len(bot.errors)} парам. Ответ Deriv (первые 3):")
        for s, e in list(bot.errors.items())[:3]:
            print(f"  {s}: {e}")
    if a.scan:
        return
    if not bot.pays:
        raise SystemExit("Deriv не дал выплату ни по одной паре (рынок закрыт?). Попробуйте позже.")
    print(f"\nРаботает на ДЕМО: {len(bot.pairs)} пар, ставка {a.stake}, лимит потерь {a.max_loss}. Ctrl+C — стоп.")
    tg_notify.send(f"▶️ Deriv-бот (валюты) запущен на ДЕМО: {len(bot.pairs)} пар, ставка {a.stake}, срок {a.minutes} мин.\n"
                   "Кнопки внизу: 💰 Баланс, 📈 Текущие ставки, 📜 История.")
    bot.k = a.k
    bot.watch = [x for x in symbols if x in bot.pays]                # следим за всеми парами, где Deriv вообще даёт ставку на этот срок
    cmds = tg_notify.Commands(tg_handlers(trader, bot=bot))
    last_pairs = last_rep = time.time()
    last_poll = 0.0
    reported = 0
    while True:
        try:
            if time.time() - last_pairs > 2 * 3600:
                bot.refresh_pairs(symbols)
                bot.watch = sorted(set(bot.watch or []) | set(bot.pays))
                bot.vol.clear()
                last_pairs = time.time()
                print(datetime.now().strftime("[%H:%M] ") + f"пары с выплатой ≥{100 * a.min_payout:.0f}%: {', '.join(bot.pairs) or 'нет'}")
            if trader.open and time.time() - last_poll >= 20:            # итоги ставок проверяем раз в 20 с (не чаще — лимиты Deriv)
                trader.poll()
                last_poll = time.time()
            if trader.halted and not trader.open:
                print("Лимит потерь достигнут, бот остановлен.\nИТОГ: " + trader.report())
                tg_notify.send("⛔ Deriv-бот остановлен по лимиту потерь.\n" + trader.report())
                return
            if not trader.halted:
                bot.step()
            getattr(trader.c, "idle_ping", lambda: None)()
            cmds.poll()
            if len(trader.done) >= reported + 25 or time.time() - last_rep > 6 * 3600:
                print(datetime.now().strftime("[%H:%M] ") + trader.report())
                tg_notify.send("📊 Deriv-бот (валюты): " + trader.report())
                reported, last_rep = len(trader.done), time.time()
        except KeyboardInterrupt:
            print("\nИТОГ: " + trader.report())
            tg_notify.send("⏹ Deriv-бот остановлен вручную.\n" + trader.report())
            return
        except Exception as e:  # noqa: BLE001
            print(f"  ошибка прохода: {type(e).__name__}: {str(e)[:120]}")
        # спим до ближайшего закрытия 5-минутной свечи (+3 с), но не дольше 20 с — чтобы и входить сразу, и проверять ставки
        to_close = GRAN - (time.time() % GRAN) + 3
        time.sleep(min(5, to_close))                                 # раз в 5 с — чтобы быстро отвечать на команды в Telegram


if __name__ == "__main__":
    main()
