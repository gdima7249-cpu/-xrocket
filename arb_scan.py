"""Сканер межбиржевого арбитража по РЕАЛЬНЫМ стаканам (ключи не нужны, ничего не покупает).

Для каждой монеты берёт стаканы со всех подключённых бирж одновременно и считает: если на бирже A купить на ARB_QUOTE USDT
(проходя по стакану, с тейкер-комиссией A), а ту же монету продать на бирже B (по стакану, с комиссией B) — сколько останется.
Показывает, как часто и насколько "чистый" разрыв бывает выше нуля.

Что НЕ учитывается: переводы между биржами (комиссия вывода, время, риск), задержка исполнения (за секунду стакан меняется),
USDT↔USD разница на Kraken/Coinbase (по умолчанию берутся только пары к USDT). Реальный арбитраж требует заранее
положить деньги на обе биржи. Поэтому положительный результат здесь — повод проверять дальше, а не готовый заработок.

    python arb_scan.py --once       # один проход по всем монетам
    python arb_scan.py              # постоянно (раз в ARB_INTERVAL секунд), статистика копится в arb_state.json
    python arb_scan.py --stats      # сводка из файла
    python arb_scan.py --selftest   # проверка расчёта на ручном примере

Настройки (.env): MULTI_EXCHANGES, MULTI_SYMBOLS (как в multi_paper.py), ARB_QUOTE=200, ARB_INTERVAL=30, ARB_ALLOW_USD=0
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import config  # noqa: F401  (подгружает .env)
from multi_paper import EXCHANGES, SYMBOLS, Venue

STATE_FILE = os.getenv("ARB_STATE", "arb_state.json")
QUOTE = float(os.getenv("ARB_QUOTE", "200"))
INTERVAL = float(os.getenv("ARB_INTERVAL", "30"))
ALLOW_USD = os.getenv("ARB_ALLOW_USD", "0") == "1"
MODEL_SLIP = 0.0025


# ---------------------------------------------------------------- расчёт
def buy_avg(asks, quote):
    """Средняя цена покупки на `quote` USDT по стакану asks=[(цена, объём)]; (количество, средняя цена) или None."""
    left, qty = quote, 0.0
    for r in asks:
        price, amount = r[0], r[1]
        take = min(left / price, amount)
        qty, left = qty + take, left - take * price
        if left <= 1e-9:
            return qty, quote / qty
    return None


def sell_proceeds(bids, qty):
    """Выручка за продажу qty по стакану bids; None, если стакана не хватает."""
    left, got = qty, 0.0
    for r in bids:
        price, amount = r[0], r[1]
        take = min(left, amount)
        got, left = got + take * price, left - take
        if left <= 1e-12:
            return got
    return None


def cross(a_book, a_fee, b_book, b_fee, quote):
    """Покупка на A, продажа на B. (чистый результат, разрыв до комиссий) в долях от quote, или None."""
    b = buy_avg(a_book["asks"], quote)
    if not b:
        return None
    qty, avg_buy = b
    got = sell_proceeds(b_book["bids"], qty * (1 - a_fee))          # комиссия покупки берётся с полученной монеты
    if got is None:
        return None
    g = sell_proceeds(b_book["bids"], qty)
    gross = g / quote - 1 if g else None
    return got * (1 - b_fee) / quote - 1, gross


# ---------------------------------------------------------------- данные
def get_book(v, sym, tickers):
    if v.modeled:                                     # у xRocket нет стакана: модель по последней цене (0.25% в каждую сторону)
        last = tickers.get(sym.replace("/", "-"))
        if not last:
            return None
        return {"asks": [(last * (1 + MODEL_SLIP), 1e12)], "bids": [(last * (1 - MODEL_SLIP), 1e12)]}
    b = v.ex.fetch_order_book(v.real(sym), 100)
    return b if b["asks"] and b["bids"] else None


def usable(v, sym):
    real = v.real(sym)
    return real.endswith("/USDT") or ALLOW_USD


def scan_symbol(venues, sym, state, tickers):
    avail = [v for v in venues if sym in v.sym_map and usable(v, sym)]
    if len(avail) < 2:
        return None
    with ThreadPoolExecutor(max_workers=len(avail)) as pool:
        futs = {v.name: pool.submit(get_book, v, sym, tickers) for v in avail}
    books = {}
    for name, f in futs.items():
        try:
            b = f.result()
            if b:
                books[name] = b
        except Exception:  # noqa: BLE001  — одна биржа не должна ломать проход
            pass
    fee = {v.name: v.fee(sym) for v in avail}
    best = None
    for a in books:
        for b in books:
            if a == b:
                continue
            r = cross(books[a], fee[a], books[b], fee[b], QUOTE)
            if not r:
                continue
            net, gross = r
            k = state["pairs"].setdefault(f"{a}->{b}", {"n": 0, "pos": 0, "sum": 0.0, "gross": 0.0, "max": -9.0})
            k["n"] += 1; k["pos"] += net > 0; k["sum"] += net; k["gross"] += gross or 0.0; k["max"] = max(k["max"], net)
            if best is None or net > best[0]:
                best = (net, gross, a, b)
    if best:
        net, gross, a, b = best
        s = state["symbols"].setdefault(sym, {"n": 0, "pos": 0, "sum": 0.0, "max": -9.0})
        s["n"] += 1; s["pos"] += net > 0; s["sum"] += net; s["max"] = max(s["max"], net)
        state["samples"] += 1
        state["pos_samples"] += net > 0
        if net > 0:
            state["found"].append({"t": time.time(), "sym": sym, "buy": a, "sell": b, "net": net, "gross": gross})
            state["found"] = state["found"][-200:]
    return best


def load():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"pairs": {}, "symbols": {}, "samples": 0, "pos_samples": 0, "found": [], "started": time.time()}


def stats(s):
    days = (time.time() - s["started"]) / 86400
    out = [f"Сканер арбитража: работает {days:.2f} дн., проверок монет {s['samples']}, из них с чистым плюсом после комиссий "
           f"{s['pos_samples']} ({100 * s['pos_samples'] / max(1, s['samples']):.1f}%). Сумма сделки {QUOTE:.0f} USDT.",
           "", "Лучшие направления (купить на A -> продать на B), чистый результат в %:",
           f"  {'направление':<22}{'проверок':>9}{'с плюсом':>9}{'среднее%':>10}{'до комиссий%':>13}{'максимум%':>11}"]
    rows = sorted(s["pairs"].items(), key=lambda kv: -kv[1]["sum"] / max(1, kv[1]["n"]))[:12]
    for k, v in rows:
        out.append(f"  {k:<22}{v['n']:>9}{v['pos']:>9}{100 * v['sum'] / v['n']:>10.3f}{100 * v['gross'] / v['n']:>13.3f}{100 * v['max']:>11.3f}")
    out += ["", "По монетам (лучшее направление на каждой проверке):",
            f"  {'монета':<12}{'проверок':>9}{'с плюсом':>9}{'среднее%':>10}{'максимум%':>11}"]
    for k, v in sorted(s["symbols"].items(), key=lambda kv: -kv[1]["sum"] / max(1, kv[1]["n"]))[:12]:
        out.append(f"  {k:<12}{v['n']:>9}{v['pos']:>9}{100 * v['sum'] / v['n']:>10.3f}{100 * v['max']:>11.3f}")
    if s["found"]:
        out += ["", "Последние найденные возможности (чистый плюс):"]
        for f in s["found"][-8:]:
            out.append(f"  {datetime.fromtimestamp(f['t']):%m-%d %H:%M}  {f['sym']:<10}{f['buy']:>8} -> {f['sell']:<8}"
                       f"чистый {100 * f['net']:+.3f}%  (до комиссий {100 * (f['gross'] or 0):+.3f}%)")
    out += ["", "Помните: нужно держать деньги на обеих биржах, задержка исполнения съедает часть разрыва, переводы между"
            " биржами платные. Реальный результат будет ниже."]
    return "\n".join(out)


def selftest():
    a = {"asks": [(100.0, 10)], "bids": [(99.9, 10)]}
    b = {"asks": [(101.1, 10)], "bids": [(101.0, 10)]}
    ok = []
    net, gross = cross(a, 0.001, b, 0.001, 200)
    ok.append(("разрыв 1% при комиссиях 0.1%+0.1% даёт ≈ +0.80%", abs(net - 0.0080) < 0.0002 and abs(gross - 0.01) < 1e-9))
    net2, _ = cross(b, 0.001, a, 0.001, 200)
    ok.append(("обратное направление убыточно (≈ -1.2%)", -0.014 < net2 < -0.010))
    ok.append(("не хватает стакана -> None", cross({"asks": [(100.0, 0.1)], "bids": []}, 0.001, b, 0.001, 200) is None))
    deep = {"asks": [(100.0, 1), (102.0, 10)], "bids": [(99.9, 10)]}
    q, avg = buy_avg(deep["asks"], 200)                          # 1 монета по 100 + 100 USDT по 102
    ok.append(("покупка идёт вглубь стакана", abs(avg - 200 / (1 + 100 / 102)) < 1e-9))
    bad = 0
    for name, good in ok:
        print(("  ok   " if good else "  ОШИБКА  ") + name)
        bad += not good
    print(f"\nИтог: {len(ok) - bad}/{len(ok)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.stats:
        if not os.path.exists(STATE_FILE):
            raise SystemExit("Файла состояния ещё нет.")
        print(stats(load()))
        return
    venues = []
    for name in EXCHANGES:
        try:
            v = Venue(name)
            print(f"{name}: подключено, пар {len(v.symbols)}")
            venues.append(v)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: пропущена ({str(e)[:100]})")
    if len(venues) < 2:
        raise SystemExit("Нужно минимум две биржи.")
    state, last_beat = load(), time.time()
    if not a.once:
        print(f"Работает. Раз в {INTERVAL:.0f} с проверка всех монет; строка «живой» раз в 10 минут. Остановка: Ctrl+C.")
    while True:
        tickers = {}
        for v in venues:
            if v.modeled:
                try:
                    tickers = v.ex.fetch_tickers()
                except Exception:  # noqa: BLE001
                    tickers = {}
        for sym in SYMBOLS:
            try:
                scan_symbol(venues, sym, state, tickers)
            except Exception as e:  # noqa: BLE001
                print(f"{sym}: {type(e).__name__}: {str(e)[:100]}")
        try:
            json.dump(state, open(STATE_FILE, "w"))
        except OSError as e:
            print("Не удалось сохранить состояние:", e)
        if a.once:
            print(stats(state))
            return
        if time.time() - last_beat >= 600:
            last_beat = time.time()
            print(f"[{datetime.now():%H:%M}] живой: проверок {state['samples']}, с плюсом {state['pos_samples']}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
