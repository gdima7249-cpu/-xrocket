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
from datetime import datetime, timezone

import deriv_demo_bot as B
import tg_notify

SHADOW = "deriv_shadow.csv"                                          # теневой журнал всех сигналов (без денег)
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


def payout_15m(client, currency, symbol, stake=10.0, direction=1):
    """Выплата в долях ставки для CALL (PUT при direction=-1) на MINUTES минут — на ТОЙ сумме, которую бот реально ставит."""
    pr = client.call({"proposal": 1, "amount": stake, "basis": "stake", "contract_type": "CALL" if direction > 0 else "PUT",
                      "currency": currency,
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


def is_usd(sym):
    """Пара с долларом (frxEURUSD, frxUSDJPY…). Остальные — кроссы (frxGBPNZD, frxEURAUD…): по истории отскок на них сильнее."""
    return "USD" in sym[3:]


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
        self.trend_filter = True                                     # ставить только если тренд за 12 ч вверх (флаг --no-trend-filter)
        self.skip_hours = (12, 17)                                   # часы новостей США (UTC): не ставим, только тень (флаг --skip-hours)
        self.usd_pairs = False                                       # пары с долларом: только тень (флаг --with-usd — ставить и на них)
        self.evening_hours, self.evening_x = (17, 24), 1.0           # вечер (UTC, 20:00–03:00 мск): ставка × evening_x (флаг --evening-x)
        self.only_evening = False                                    # ставить только вечером, днём — только тень (флаг --only-evening)
        self.london, self.london_hours = False, (7, 12)              # вместе с --only-evening: ставить ещё и 10:00–15:00 мск (флаг --london)
        self.evening_all = False                                     # вечером ставить и при тренде вниз: падение → вверх, взлёт → вниз
        self.shadow = []                                             # теневой журнал: ВСЕ сигналы (и поставленные, и пропущенные)
        self.shadow_path = SHADOW
        self.vol, self.last, self.pairs, self.pays = {}, {}, [], {}
        self.vol_t = {}                                              # когда обновлён разброс по паре (обновляем раз в сутки)

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
        """«Обычный разброс» 5-минутных изменений за ~неделю — БЕЗ скачков через перерывы в торговле (ночь, выходные),
        иначе разрыв пятница→понедельник раздувает разброс, и сигналов становится меньше, чем в истории."""
        cs = self.pub.candles(symbol, 2000)
        r = [math.log(b[1] / a[1]) for a, b in zip(cs, cs[1:]) if b[0] - a[0] == GRAN and a[1] > 0 and b[1] > 0]
        if len(r) > 100:
            m = sum(r) / len(r)
            self.vol[symbol] = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
            self.vol_t[symbol] = time.time()

    def settle_shadow(self):
        """Через срок ставки смотрим, куда ушла цена по каждому записанному сигналу (по закрытию 5-минутной свечи)."""
        import csv
        n_bars = MINUTES // 5
        now = time.time()
        for sh in list(self.shadow):
            exit_open = sh["t"] + n_bars * GRAN                      # свеча, закрытие которой = через срок после сигнала
            if now < exit_open + GRAN + 3:
                continue
            try:
                cs = self.pub.candles(sh["sym"], n_bars + 6)
            except Exception:  # noqa: BLE001
                continue
            ex = [c for t, c in cs if t == exit_open]
            self.shadow.remove(sh)
            if not ex or ex[0] == sh["entry"]:
                continue                                             # перерыв в торговле или ничья — не считаем
            up = sh.get("dir", 1) == 1
            win = 1 if (ex[0] > sh["entry"]) == up else 0
            new = not os.path.exists(self.shadow_path)
            with open(self.shadow_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["время", "пара", "сила", "вход", "выход", "выиграл", "правило", "тренд12ч", "время_суток"])
                w.writerow([datetime.fromtimestamp(sh["t"] + GRAN, timezone.utc).strftime("%Y-%m-%d %H:%M"), sh["sym"],
                            round(sh["z"], 2), sh["entry"], ex[0], win, "падение→вверх" if up else "рост→вниз",
                            sh.get("trend", ""), sh.get("session", "")])

    def step(self):
        """Проверить новые закрытые свечи по всем парам. Каждый сигнал — в теневой журнал; ставка ВВЕРХ — только по свежему
        сигналу (последняя закрытая свеча, не позже 90 с после её закрытия) и если выплата ≥ порога."""
        busy = {o.get("base") for o in self.tr.open.values()}
        closed_open = (int(time.time()) // GRAN - 1) * GRAN           # время открытия последней закрытой свечи
        for s in (self.watch or self.pairs):
            if self.last.get(s) == closed_open:                     # эту свечу уже проверили — не тратим запрос
                continue
            try:
                if s not in self.vol or time.time() - self.vol_t.get(s, 0) > 86400:
                    self.refresh_vol(s)
                if s not in self.vol:
                    continue
                cs = self.pub.candles(s, 12)
            except Exception as e:  # noqa: BLE001
                self.log(f"  {s}: нет данных ({type(e).__name__}: {str(e)[:60]}), проверю на следующем проходе")
                continue
            if len(cs) < 4:
                continue
            prev = self.last.get(s)
            self.last[s] = cs[-1][0]
            # все свечи, которые ещё не проверяли (если бот отстал — несколько)
            idx = [j for j in range(3, len(cs)) if prev is None and j == len(cs) - 1 or prev is not None and cs[j][0] > prev]
            for j in idx:
                win4 = cs[j - 3:j + 1]
                if any(y[0] - x[0] != GRAN for x, y in zip(win4, win4[1:])):   # внутри окна перерыв в торговле
                    continue
                sig, z = is_signal([c for _, c in win4], self.vol[s], self.k)
                hour = ((cs[j][0] + GRAN) % 86400) // 3600              # час закрытия сигнальной свечи (UTC)
                news = bool(self.skip_hours) and self.skip_hours[0] <= hour < self.skip_hours[1]
                evening = self.evening_hours[0] <= hour < self.evening_hours[1]
                fresh = j == len(cs) - 1 and time.time() - (cs[j][0] + GRAN) < 90
                london = self.london_hours[0] <= hour < self.london_hours[1]                                # Лондон до открытия Нью-Йорка: 10:00–15:00 мск
                session = "новости" if news else "вечер" if evening else "лондон" if london else "спокойно"
                if z >= self.k:                                     # резкий ВЗЛЁТ: в тень; вечером при тренде вниз (--evening-all) — ставка ВНИЗ
                    trp = self.trend12(s, cs[j][0])
                    self.shadow.append({"sym": s, "t": cs[j][0], "entry": cs[j][1], "z": z, "dir": -1,
                                        "trend": "вверх" if trp is not None and trp > 0 else "вниз" if trp is not None else "?",
                                        "session": session})
                    self.cnt["взлётов (только тень)"] = self.cnt.get("взлётов (только тень)", 0) + 1
                    if (self.evening_all and evening and fresh and not news and trp is not None and trp < 0
                            and (self.usd_pairs or not is_usd(s))):
                        self.cnt["вечерних взлётов → ставка вниз"] = self.cnt.get("вечерних взлётов → ставка вниз", 0) + 1
                        self._maybe_bet(s, z, cs[j][1], busy, self.evening_x, direction=-1)
                if not sig:
                    continue
                self.cnt["сигналов"] += 1
                tr = self.trend12(s, cs[j][0])
                self.shadow.append({"sym": s, "t": cs[j][0], "entry": cs[j][1], "z": z, "dir": 1,
                                    "trend": "вверх" if tr is not None and tr > 0 else "вниз" if tr is not None else "?",
                                    "session": session})
                if not fresh:
                    self.log(f"  СИГНАЛ {s} ({z:.1f}) обнаружен с опозданием — только в теневой журнал")
                    self.cnt["пропуск: опоздание"] = self.cnt.get("пропуск: опоздание", 0) + 1
                    continue
                evening_dn = self.evening_all and evening and tr is not None and tr < 0   # вечер: падение и при тренде вниз
                if self.trend_filter and not (tr is not None and tr > 0) and not evening_dn:
                    self.log(f"  СИГНАЛ {s} ({z:.1f}), но тренд за 12 ч не вверх — только теневой журнал")
                    self.cnt["пропуск: тренд не вверх"] = self.cnt.get("пропуск: тренд не вверх", 0) + 1
                    continue
                if not self.usd_pairs and is_usd(s):
                    self.log(f"  СИГНАЛ {s} ({z:.1f}), но это пара с долларом (по истории ~55%) — только теневой журнал")
                    self.cnt["пропуск: пара с долларом"] = self.cnt.get("пропуск: пара с долларом", 0) + 1
                    continue
                if news:
                    self.log(f"  СИГНАЛ {s} ({z:.1f}) в часы новостей США ({hour}:00 UTC) — только теневой журнал")
                    self.cnt["пропуск: часы новостей"] = self.cnt.get("пропуск: часы новостей", 0) + 1
                    continue
                if self.only_evening and not evening and not (self.london and london):
                    self.log(f"  СИГНАЛ {s} ({z:.1f}) днём ({hour}:00 UTC), ставим только вечером — только теневой журнал")
                    self.cnt["пропуск: не вечер"] = self.cnt.get("пропуск: не вечер", 0) + 1
                    continue
                self._maybe_bet(s, z, cs[j][1], busy, self.evening_x if evening else 1.0)

    def trend12(self, s, t_sig):
        """Тренд за ~12 часов ДО резкого падения: log(close[t-3] / close[t-144]) по 5-минутным свечам, без перерывов в торговле.
        Возвращает число (>0 — рынок рос) или None, если данных нет или внутри окна был перерыв."""
        try:
            cs = self.pub.candles(s, 160)
        except Exception:  # noqa: BLE001
            return None
        idx = [i for i, c in enumerate(cs) if c[0] == t_sig]
        if not idx or idx[0] < 144:
            return None
        j = idx[0]
        win = cs[j - 144:j + 1]
        if any(b[0] - a[0] != GRAN for a, b in zip(win, win[1:])):
            return None
        return math.log(cs[j - 3][1] / cs[j - 144][1])

    def _maybe_bet(self, s, z, price, busy, mult=1.0, direction=1):
        if s in busy:
            self.log(f"  СИГНАЛ {s} ({z:.1f} разброса), но по паре уже есть открытая ставка — пропуск")
            self.cnt["пропуск: пара занята"] += 1
            return
        if self.max_open and len(self.tr.open) >= self.max_open:
            self.log(f"  СИГНАЛ {s} ({z:.1f}), но открыто уже {len(self.tr.open)} ставок (лимит {self.max_open}) — пропуск")
            self.cnt["пропуск: лимит открытых"] += 1
            return
        try:
            p = payout_15m(self.tr.c, self.tr.currency, s, self.tr.stake, direction)
        except Exception as e:  # noqa: BLE001
            self.log(f"  СИГНАЛ {s}, но цену ставки получить не удалось: {str(e)[:80]}")
            return
        if p < self.min_payout:
            self.log(f"  СИГНАЛ {s} ({z:.1f}), но выплата сейчас {100 * p:.0f}% < {100 * self.min_payout:.0f}% — пропуск")
            self.cnt["пропуск: низкая выплата"] += 1
            self.low_pays.append(p)
            return
        if self.stake_pct:                                      # ставка = доля текущего баланса, в пределах [мин, макс]
            try:
                bal = float(self.tr.c.call({"balance": 1})["balance"]["balance"])
                self.tr.stake = round(min(self.stake_max, max(self.stake_min, self.stake_pct * bal)), 2)
            except Exception:  # noqa: BLE001
                self.tr.stake = self.stake_min
        base = self.tr.stake
        if mult != 1.0:                                         # вечером ставка крупнее (только на эту ставку)
            self.tr.stake = round(base * mult, 2)
        self.log(f"  СИГНАЛ {s}: {'падение' if direction > 0 else 'взлёт'} {abs(z):.1f} разброса за 15 мин, выплата {100 * p:.0f}% "
                 f"→ {'ВВЕРХ' if direction > 0 else 'ВНИЗ'} на {MINUTES} мин, "
                 f"ставка {self.tr.stake}" + (f" (вечер ×{mult:g})" if mult != 1.0 else ""))
        try:
            if self.tr.place(s, direction, price, symbol=s):
                self.cnt["поставлено"] += 1
        except Exception as e:  # noqa: BLE001
            self.log(f"  {s}: ставка не прошла: {str(e)[:120]}")
            self.cnt["ошибка ставки"] += 1
        finally:
            self.tr.stake = base


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


def shadow_text(path):
    import csv
    if not os.path.exists(path):
        return "\n\n👁 Теневой журнал (все сигналы, без денег): пока пуст."
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    if not rows:
        return "\n\n👁 Теневой журнал: пока пуст."

    def part(rs, name, expect):
        n, w = len(rs), sum(r[5] == "1" for r in rs)
        if not n:
            return f"\n  {name}: пока нет сигналов."
        p = w / n
        z2 = 1.96 ** 2
        lo = (p + z2 / (2 * n) - 1.96 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)
        hi = (p + z2 / (2 * n) + 1.96 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)
        return f"\n  {name}: {n} сигналов, угадано {w} ({100 * p:.1f}%, 95%-интервал {100 * lo:.1f}…{100 * hi:.1f}%). {expect}"

    down = [r for r in rows if len(r) < 7 or r[6] == "падение→вверх"]
    pump = [r for r in rows if len(r) >= 7 and r[6] == "рост→вниз"]
    d_up = [r for r in down if len(r) >= 8 and r[7] == "вверх"]
    d_dn = [r for r in down if len(r) >= 8 and r[7] == "вниз"]
    return ("\n\n👁 Теневой журнал (ВСЕ сигналы, проверка без денег):"
            + part(down, "падение → вверх, все", "По истории ~57%.")
            + part(d_up, "   из них тренд 12ч ВВЕРХ (бот ставит)", "По истории ~60%.")
            + part(d_dn, "   из них тренд 12ч вниз (без ставок)", "По истории ~54%.")
            + part([r for r in d_up if len(r) >= 9 and r[8] in ("спокойно", "вечер", "лондон")], "      тренд вверх, вне часов новостей",
                   "По истории ~66%.")
            + part([r for r in d_up if len(r) >= 9 and r[8] == "вечер"], "         из них вечер 20–03 мск (ставка крупнее)",
                   "По истории ~67% (без перерыва Deriv 23:50–01:00 мск).")
            + part([r for r in d_up if len(r) >= 9 and r[8] == "лондон" and not is_usd(r[1])],
                   "         из них 10–15 мск, кроссы (ставит с --london)", "По истории ~61%, нужно ~55%.")
            + part([r for r in d_up if len(r) >= 9 and r[8] != "новости" and not is_usd(r[1])],
                   "      тренд вверх, вне новостей, КРОССЫ (бот ставит)", "По истории кроссы ~62% (все сигналы).")
            + part([r for r in d_up if len(r) >= 9 and r[8] != "новости" and is_usd(r[1])],
                   "      тренд вверх, вне новостей, пары с $ (без ставок)", "По истории ~55% (все сигналы).")
            + part([r for r in d_up if len(r) >= 9 and r[8] == "новости"], "      тренд вверх, 15–20 мск (новости, без ставок)",
                   "По истории ~51%.")
            + part([r for r in d_dn if len(r) >= 9 and r[8] == "вечер" and not is_usd(r[1])],
                   "   падение, тренд вниз, ВЕЧЕР, кроссы (ставит с --evening-all)", "Без перерыва Deriv по истории ~50%.")
            + part(pump, "рост → вниз, все (только наблюдение)", "По истории ~54%.")
            + part([r for r in pump if len(r) >= 9 and r[7] == "вниз" and r[8] == "вечер" and not is_usd(r[1])],
                   "   взлёт, тренд вниз, ВЕЧЕР, кроссы (ставит ВНИЗ с --evening-all)", "Без перерыва Deriv по истории ~58%.")
            + part([r for r in pump if len(r) >= 8 and r[7] == "вниз"], "   из них тренд 12ч ВНИЗ", "Нужно 57%+.")
            + part([r for r in pump if len(r) >= 8 and r[7] == "вверх"], "   из них тренд 12ч вверх", ""))


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
            ", тренд не вверх " + str(c.get("пропуск: тренд не вверх", 0)) +
            ", часы новостей " + str(c.get("пропуск: часы новостей", 0)) +
            ", не вечер " + str(c.get("пропуск: не вечер", 0)) +
            "; вечерних взлётов → ставка вниз " + str(c.get("вечерних взлётов → ставка вниз", 0)) +
            ", пара с долларом " + str(c.get("пропуск: пара с долларом", 0)) +
            ", опоздание " + str(c.get("пропуск: опоздание", 0)) + ", ошибки " + str(c["ошибка ставки"]) +
            "; резких взлётов (только тень): " + str(c.get("взлётов (только тень)", 0)) +
            f". Под наблюдением пар: {len(bot.watch or bot.pairs)}, из них сейчас с выплатой ≥ порога: {len(bot.pairs)}."
            + shadow_text(bot.shadow_path))


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
        rows = [f"{o['base'].replace('frx', '')}: {'ВВЕРХ' if o.get('dir', 1) > 0 else 'ВНИЗ'}, ставка {o['stake']}, выплата {o['payout']:.2f}, "
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
    now = int(time.time()) - 5                                      # последняя свеча закрылась 5 с назад («свежий» сигнал)

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
    bot.trend_filter, bot.skip_hours = False, None   # старые тесты — без фильтров
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
    gbot.trend_filter, gbot.skip_hours = False, None   # старые тесты — без фильтров
    gbot.refresh_pairs(["frxGAP"])
    gbot.vol["frxGAP"] = vol
    gbot.step()
    ok.append(("падение через перерыв в торговле (выходные) не даёт ставки", not gtr.open))
    low = FxBot(pub, B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m"), 0.95, log=msgs.append)
    low.trend_filter, low.skip_hours = False, None   # старые тесты — без фильтров
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
    cbot.trend_filter, cbot.skip_hours = False, None   # старые тесты — без фильтров
    cbot.refresh_pairs(["frxAAA"])
    cbot.step()
    ok.append(("сложный процент: ставка 2% от баланса 40 = 0.8", len(ctr.open) == 1 and abs(ctr.stake - 0.8) < 1e-9))

    # бот отстал: сигнал был в предыдущей свече, а последняя закрытая — уже следующая → только теневой журнал, без ставки
    lag_pub = FakePub()
    lag_pub.data["frxLAG"] = base + [(now - 5 * GRAN, closes[-1]), (now - 4 * GRAN, crash[-3]), (now - 3 * GRAN, crash[-2]),
                                     (now - 2 * GRAN, crash[-1]), (now - GRAN, closes[-1])]   # последняя свеча — уже отскок
    ltr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    lbot = FxBot(lag_pub, ltr, 0.85, log=msgs.append)
    lbot.trend_filter, lbot.skip_hours = False, None   # старые тесты — без фильтров
    lbot.refresh_pairs(["frxLAG"])
    lbot.vol["frxLAG"] = vol
    lbot.vol_t["frxLAG"] = time.time()
    lbot.last["frxLAG"] = now - 5 * GRAN                            # последняя проверенная — до падения
    lbot.step()
    ok.append(("бот отстал: старый сигнал записан в теневой журнал, но без ставки", lbot.cnt["сигналов"] >= 1
               and not ltr.open and lbot.cnt.get("пропуск: опоздание", 0) >= 1))

    # фильтр тренда: 12 ч рост, затем резкое падение → ставка; 12 ч спад, затем падение → только тень
    def series(trend_up):
        pts = []
        for i in range(200):
            drift = (1 if trend_up else -1) * 0.00004 * i
            pts.append((now - (204 - i) * GRAN, 1.0 * math.exp(drift + 0.0001 * ((i * 7) % 5 - 2))))
        last = pts[-1][1]
        pts += [(now - 4 * GRAN, last), (now - 3 * GRAN, last * (1 - 3 * vol)), (now - 2 * GRAN, last * (1 - 6 * vol)),
                (now - GRAN, last * (1 - 9 * vol))]
        return pts

    for up, name in ((True, "UP"), (False, "DN")):
        tp = FakePub()
        tp.data["frx" + name] = series(up)
        ttr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
        tb = FxBot(tp, ttr, 0.85, log=msgs.append)
        tb.skip_hours = None
        tb.refresh_pairs(["frx" + name])
        tb.vol["frx" + name] = vol
        tb.vol_t["frx" + name] = time.time()
        tb.step()
        if up:
            ok.append(("фильтр тренда: 12 ч рост + резкое падение → ставка", len(ttr.open) == 1
                       and tb.shadow and tb.shadow[-1].get("trend") == "вверх"))
        else:
            ok.append(("фильтр тренда: 12 ч спад + резкое падение → без ставки, только тень", not ttr.open
                       and tb.shadow and tb.shadow[-1].get("trend") == "вниз"
                       and tb.cnt.get("пропуск: тренд не вверх", 0) == 1))

    # часы новостей: тот же сигнал, но окно «часов новостей» покрывает текущий час → без ставки, только тень
    hp = FakePub()
    hp.data["frxUP"] = series(True)
    htr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    hb = FxBot(hp, htr, 0.85, log=msgs.append)
    hb.skip_hours = (0, 24)
    hb.refresh_pairs(["frxUP"])
    hb.vol["frxUP"] = vol
    hb.vol_t["frxUP"] = time.time()
    hb.step()
    ok.append(("часы новостей: без ставки, сигнал в тени с пометкой «новости»", not htr.open and hb.shadow
               and hb.shadow[-1].get("session") == "новости" and hb.cnt.get("пропуск: часы новостей", 0) == 1))

    # вечер: тот же сигнал, окно «вечера» покрывает текущий час → ставка ×2, потом базовая ставка возвращается
    ep = FakePub()
    ep.data["frxUP"] = series(True)
    etr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    eb = FxBot(ep, etr, 0.85, log=msgs.append)
    eb.skip_hours, eb.evening_hours, eb.evening_x = None, (0, 24), 2.0
    eb.refresh_pairs(["frxUP"])
    eb.vol["frxUP"] = vol
    eb.vol_t["frxUP"] = time.time()
    eb.step()
    ok.append(("вечер: ставка ×2 (2.0 вместо 1), после ставки снова 1, в тени пометка «вечер»", len(etr.open) == 1
               and abs(next(iter(etr.open.values())).get("stake", 2.0) - 2.0) < 1e-9 and etr.stake == 1
               and eb.shadow[-1].get("session") == "вечер"))

    # только вечер: окно «вечера» не покрывает текущий час → без ставки, только тень
    op = FakePub()
    op.data["frxUP"] = series(True)
    otr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    ob = FxBot(op, otr, 0.85, log=msgs.append)
    ob.skip_hours, ob.evening_hours, ob.only_evening = None, (0, 0), True
    ob.refresh_pairs(["frxUP"])
    ob.vol["frxUP"] = vol
    ob.vol_t["frxUP"] = time.time()
    ob.step()
    ok.append(("только вечер: днём без ставки, только тень", not otr.open and ob.shadow
               and ob.cnt.get("пропуск: не вечер", 0) == 1))

    # вечер + --evening-all: падение при тренде вниз → ставка ВВЕРХ; взлёт при тренде вниз → ставка ВНИЗ
    def pump_series():
        pts = [(t, c) for t, c in series(False)[:-4]]
        last = pts[-1][1]
        return pts + [(now - 4 * GRAN, last), (now - 3 * GRAN, last * (1 + 3 * vol)), (now - 2 * GRAN, last * (1 + 6 * vol)),
                      (now - GRAN, last * (1 + 9 * vol))]

    for name, data, want in (("frxDN", series(False), 1), ("frxPU", pump_series(), -1)):
        ap_ = FakePub()
        ap_.data[name] = data
        atr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
        ab = FxBot(ap_, atr, 0.85, log=msgs.append)
        ab.skip_hours, ab.evening_hours, ab.evening_all = None, (0, 24), True
        ab.refresh_pairs([name])
        ab.vol[name] = vol
        ab.vol_t[name] = time.time()
        ab.step()
        dirs = [o.get("dir") for o in atr.open.values()]
        ok.append((f"вечер, тренд вниз, {'падение → ВВЕРХ' if want > 0 else 'взлёт → ВНИЗ'} (--evening-all)", dirs == [want]))
        if want > 0:                                                 # без --evening-all тот же сигнал не ставится
            ap2 = FakePub()
            ap2.data[name] = data
            atr2 = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
            ab2 = FxBot(ap2, atr2, 0.85, log=msgs.append)
            ab2.skip_hours, ab2.evening_hours = None, (0, 24)
            ab2.refresh_pairs([name])
            ab2.vol[name] = vol
            ab2.vol_t[name] = time.time()
            ab2.step()
            ok.append(("без --evening-all падение при тренде вниз не ставится", not atr2.open))

    # --london: вне вечера, но в «лондонские» часы — ставка; без --london — только тень
    for lon in (True, False):
        lp = FakePub()
        lp.data["frxUP"] = series(True)
        ltr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
        lb = FxBot(lp, ltr, 0.85, log=msgs.append)
        lb.skip_hours, lb.evening_hours, lb.only_evening = None, (0, 0), True
        lb.london, lb.london_hours = lon, (0, 24)
        lb.refresh_pairs(["frxUP"])
        lb.vol["frxUP"] = vol
        lb.vol_t["frxUP"] = time.time()
        lb.step()
        ok.append((f"--london {'вкл' if lon else 'выкл'}: {'ставка' if lon else 'только тень'}, в тени пометка «лондон»",
                   len(ltr.open) == (1 if lon else 0) and lb.shadow[-1].get("session") == "лондон"))

    # пара с долларом: тот же сигнал → без ставки, только тень
    up_ = FakePub()
    up_.data["frxEURUSD"] = series(True)
    utr = B.Trader(B.FakeClient(), 1, 15, 10 ** 9, 20, log=lambda *a: None, unit="m")
    ub = FxBot(up_, utr, 0.85, log=msgs.append)
    ub.skip_hours = None
    ub.refresh_pairs(["frxEURUSD"])
    ub.vol["frxEURUSD"] = vol
    ub.vol_t["frxEURUSD"] = time.time()
    ub.step()
    ok.append(("пара с долларом: без ставки, только тень; кроссы и доллар различаются", not utr.open and ub.shadow
               and ub.cnt.get("пропуск: пара с долларом", 0) == 1 and is_usd("frxUSDJPY") and not is_usd("frxGBPNZD")))

    # теневой журнал: сигнал записан; через 30 мин по свече выхода определяется итог
    global MINUTES
    MINUTES = 30
    import tempfile as _tf
    bot.shadow_path = os.path.join(_tf.mkdtemp(), "sh.csv")
    sh0 = dict(bot.shadow[0]) if bot.shadow else None
    if sh0:
        exit_open = sh0["t"] + 6 * GRAN
        pub.data["frxAAA"] = pub.data["frxAAA"] + [(exit_open, sh0["entry"] * 1.001)]
        bot.shadow = [sh0]
        real_time = time.time
        time.time = lambda: exit_open + GRAN + 10
        try:
            bot.settle_shadow()
        finally:
            time.time = real_time
    ok.append(("теневой журнал: сигнал записан и через срок засчитан как выигрыш", bool(sh0) and
               "1 сигналов" in shadow_text(bot.shadow_path).replace("ов, цена", "ов, цена") and "100.0%" in shadow_text(bot.shadow_path)))
    MINUTES = 15

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
    ap.add_argument("--skip-hours", default="12-17", help="часы (UTC), когда не ставить: «12-17» = 15:00–20:00 мск; «none» — ставить всегда")
    ap.add_argument("--with-usd", action="store_true",
                    help="ставить и на пары с долларом (EURUSD, USDJPY…). По умолчанию на них только теневой журнал: по истории ~55%%")
    ap.add_argument("--only-evening", action="store_true",
                    help="ставить только вечером (20:00–03:00 мск); дневные сигналы — только в теневой журнал")
    ap.add_argument("--london", action="store_true",
                    help="вместе с --only-evening ставить ещё и 10:00–15:00 мск (по истории ~61%%, на грани — для проверки на демо)")
    ap.add_argument("--evening-all", action="store_true",
                    help="вечером ставить и при тренде 12 ч вниз: падение → ВВЕРХ, взлёт → ВНИЗ. Без перерыва Deriv по истории ~50%% и ~58%% — НЕ советую")
    ap.add_argument("--evening-x", type=float, default=1.0,
                    help="вечером (20:00–03:00 мск) ставка во столько раз больше, например 2. По умолчанию 1 — как днём")
    ap.add_argument("--no-trend-filter", action="store_true", help="ставить на все падения, без фильтра «тренд за 12 ч вверх»")
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
        # рынок закрыт (выходные): не выходим — ждём открытия, всё это время отвечаем на кнопки в Telegram
        msg = "Рынок закрыт (Deriv не даёт ставок). Жду открытия, проверяю раз в 10 минут."
        if bot.errors:
            msg += " Ответ Deriv: " + next(iter(bot.errors.values()))[:120]
        print(msg)
        tg_notify.send("⏸ " + msg)
        wait_cmds = tg_notify.Commands(tg_handlers(trader, bot=bot))
        last_try = time.time()
        while not bot.pays:
            try:
                wait_cmds.poll()
                getattr(trader.c, "idle_ping", lambda: None)()
                if time.time() - last_try > 600:
                    bot.refresh_pairs(symbols)
                    last_try = time.time()
            except KeyboardInterrupt:
                return
            except Exception as e:  # noqa: BLE001
                print(f"  ожидание: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(5)
        print(datetime.now().strftime("[%H:%M] ") + "рынок открылся.")
    print(f"\nРаботает на ДЕМО: {len(bot.pairs)} пар, ставка {a.stake}, лимит потерь {a.max_loss}. Ctrl+C — стоп.")
    tg_notify.send(f"▶️ Deriv-бот (валюты) запущен на ДЕМО: {len(bot.pairs)} пар, ставка {a.stake}, срок {a.minutes} мин.\n"
                   "Кнопки внизу: 💰 Баланс, 📈 Текущие ставки, 📜 История.")
    bot.k = a.k
    bot.trend_filter = not a.no_trend_filter
    bot.skip_hours = None if a.skip_hours.lower() == "none" else tuple(int(x) for x in a.skip_hours.split("-"))
    bot.evening_x = a.evening_x
    bot.only_evening = a.only_evening
    bot.london = a.london
    bot.evening_all = a.evening_all
    bot.usd_pairs = a.with_usd
    bot.watch = [x for x in symbols if x in bot.pays]                # следим за всеми парами, где Deriv вообще даёт ставку на этот срок
    cmds = tg_notify.Commands(tg_handlers(trader, bot=bot))
    last_pairs = last_rep = time.time()
    last_poll = 0.0
    reported = 0
    while True:
        try:
            if time.time() - last_pairs > (600 if bot.errors else 2 * 3600):   # пары без выплаты (перерыв Deriv) — проверяем чаще
                bot.refresh_pairs(symbols)
                bot.watch = sorted(set(bot.watch or []) | set(bot.pays))
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
            bot.settle_shadow()
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
