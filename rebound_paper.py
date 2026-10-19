"""Бумажная проверка ВПЕРЁД правила «покупка после резкого падения, выход через 1 час» — на живых данных, без денег и без ключей.

Правила заморожены заранее (по результатам rebound_research.py на последнем годе и на двух годах до него):
  A: падение ≥2% за одну 5-минутную свечу      → покупка по открытию следующей свечи, продажа по закрытию через 1 час (12 свечей)
  B: падение ≥3% за три 5-минутные свечи (15м) → то же
Сделки по одной монете в рамках одного правила не перекрываются. Издержки в отчёте: 0.12% на сделку (0.04% комиссии и 0.02%
проскальзывания за сторону), как в исследовании.

Раз в минуту берёт публичные 5-минутные свечи Binance по монетам, обрабатывает каждую закрытую свечу один раз, ведёт виртуальные
позиции и журнал rebound_paper.csv; состояние — rebound_paper.json (можно перезапускать, продолжит с того же места).
Никаких ордеров не отправляет.

  python rebound_paper.py                 # работать постоянно (удобно в tmux на сервере)
  python rebound_paper.py --report        # только показать итоги по журналу
  python rebound_paper.py --selftest      # проверка логики на имитации
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone

API = "https://data-api.binance.vision/api/v3/klines"
COINS = "BTC,ETH,SOL,XRP,DOGE,BNB,ADA,AVAX,LINK,LTC"
BAR = 300                                                         # секунд в 5-минутной свече
HOLD = 12                                                         # свечей удержания (1 час)
COST = 0.0012
RULES = {"A": (1, 0.02), "B": (3, 0.03)}                         # имя: (окно в свечах, порог падения)
STATE, LOG = "rebound_paper.json", "rebound_paper.csv"


def fetch(symbol, limit=40):
    import requests
    r = requests.get(API, params={"symbol": symbol, "interval": "5m", "limit": limit}, timeout=20)
    r.raise_for_status()
    return [(int(x[0]) // 1000, float(x[1]), float(x[4])) for x in r.json()]   # (время открытия, open, close)


def step(st, coin, bars, now, log):
    """Обрабатывает новые закрытые свечи монеты. bars — список (время открытия, open, close) по возрастанию, последний может быть
    незакрытым. st — состояние: {"last": {монета: время последней обработанной закрытой свечи}, "pos": {...}, "closed": [...]}."""
    closed = [b for b in bars if b[0] + BAR <= now]
    by_t = {b[0]: b for b in bars}
    last = st["last"].get(coin)
    for i, (t, o, c) in enumerate(closed):
        if last is not None and t <= last:
            continue
        # 1) выход по открытым позициям: свеча выхода — та, что открылась через HOLD-1 свечей после входа
        for rule in RULES:
            key = f"{coin}:{rule}"
            p = st["pos"].get(key)
            if p and p.get("entry") is not None and t == p["t_entry"] + (HOLD - 1) * BAR:
                ret = c / p["entry"] - 1
                rec = {"монета": coin, "правило": rule, "вход_время": p["t_entry"], "вход": p["entry"], "выход": c,
                       "доходность": ret, "после_издержек": ret - COST}
                st["closed"].append(rec)
                del st["pos"][key]
                log(f"  ВЫХОД {coin} [{rule}] {p['entry']:.6g} → {c:.6g}: {100 * ret:+.2f}% (после издержек {100 * (ret - COST):+.2f}%)")
                _write(rec)
        # 2) новый сигнал на этой закрытой свече (вход — по открытию следующей)
        for rule, (k, thr) in RULES.items():
            key = f"{coin}:{rule}"
            if key in st["pos"] or i < k:
                continue
            move = c / closed[i - k][2] - 1
            if move <= -thr:
                st["pos"][key] = {"t_entry": t + BAR, "entry": None, "signal_move": move}
                log(f"  СИГНАЛ {coin} [{rule}]: {100 * move:.2f}% за {k * 5} мин — виртуальная покупка по открытию следующей свечи")
        last = t
    st["last"][coin] = last
    # 3) цена входа появляется, как только открылась следующая свеча
    for rule in RULES:
        p = st["pos"].get(f"{coin}:{rule}")
        if p and p["entry"] is None and p["t_entry"] in by_t:
            p["entry"] = by_t[p["t_entry"]][1]
            log(f"  ВХОД {coin} [{rule}] по {p['entry']:.6g}")


def _write(rec):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(list(rec))
        w.writerow([datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d %H:%M") if k == "вход_время" else v for k, v in rec.items()])


def report(closed):
    if not closed:
        return "Пока нет закрытых сделок."
    lines = []
    for rule in ["все"] + list(RULES):
        xs = [r["после_издержек"] for r in closed if rule == "все" or r["правило"] == rule]
        if not xs:
            continue
        n = len(xs)
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        w = sum(x > 0 for x in xs)
        lines.append(f"  [{rule}] сделок {n}, в плюсе {100 * w / n:.0f}%, в среднем {100 * m:+.3f}% после издержек "
                     f"(± {200 * se:.3f}%), сумма {100 * sum(xs):+.2f}% от ставки")
    lines.append("  Ожидание по истории: правило A около +0.5%, правило B около +0.5% на сделку. Для вывода нужно 100+ сделок"
                 " (это месяцы: сигналы идут кучами в дни обвалов).")
    return "\n".join(lines)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"last": {}, "pos": {}, "closed": []}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, STATE)


def run(coins):
    st = load_state()
    print(f"Бумажная проверка: {', '.join(coins)}; правила A (≥2% за 5м) и B (≥3% за 15м), выход через 1 ч. Ордеров нет. Ctrl+C — стоп.")
    print(report(st["closed"]))
    last_report = time.time()
    while True:
        now = int(time.time())
        for coin in coins:
            try:
                step(st, coin, fetch(coin + "USDT"), now, print)
            except Exception as e:  # noqa: BLE001
                print(f"  {coin}: ошибка загрузки ({type(e).__name__}: {str(e)[:80]}), попробую через минуту")
        save_state(st)
        if time.time() - last_report > 6 * 3600:
            print(datetime.now().strftime("[%Y-%m-%d %H:%M]"), "итоги:\n" + report(st["closed"]))
            last_report = time.time()
        time.sleep(60)


def selftest():
    ok = 0
    # имитация: ровная цена, затем падение на 2.5% за одну свечу, затем рост на 1% за час
    t0 = 1_700_000_000 // BAR * BAR
    prices = [100.0] * 30 + [97.5] + [97.5 * (1 + 0.01 * j / 12) for j in range(1, 20)]
    bars = []
    for j, p in enumerate(prices):
        o = prices[j - 1] if j else p
        bars.append((t0 + j * BAR, o, p))
    st, msgs = {"last": {}, "pos": {}, "closed": []}, []
    global _write
    real_write, _write = _write, lambda rec: None
    try:
        # подаём свечи по одной, как в реальной работе (каждая — сначала незакрытой, потом закрытой)
        for j in range(25, len(bars)):
            step(st, "TST", bars[max(0, j - 39):j + 1], bars[j][0] + BAR - 1, msgs.append)   # последняя ещё не закрыта
            step(st, "TST", bars[max(0, j - 39):j + 1], bars[j][0] + BAR, msgs.append)       # теперь закрыта
    finally:
        _write = real_write
    a = [r for r in st["closed"] if r["правило"] == "A"]
    checks = [
        ("сигнал A сработал на падении 2.5%", any("СИГНАЛ TST [A]" in m for m in msgs)),
        ("сигнал B не сработал (падение за 15м меньше 3%)", not any("СИГНАЛ TST [B]" in m for m in msgs)),
        ("вход по открытию следующей свечи (97.5)", bool(a) and abs(a[0]["вход"] - 97.5) < 1e-9),
        ("выход через 12 свечей с доходностью ≈ +1%", bool(a) and abs(a[0]["доходность"] - 0.01) < 1e-9),
        ("каждая свеча обработана один раз (одна сделка)", len(a) == 1),
        ("отчёт строится", "сделок 1" in report(st["closed"])),
    ]
    for name, good in checks:
        print(("  ok   " if good else "  FAIL ") + name)
        ok += good
    print(f"\nИтог: {ok}/{len(checks)} проверок пройдено")
    return ok == len(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default=COINS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.report:
        print(report(load_state()["closed"]))
        return
    run(a.coins.split(","))


if __name__ == "__main__":
    main()
