"""Бот для ДЕМО-счёта Deriv: ставит по потоку тиков (Rise/Fall на 5 тиков), сигналы даёт модель, обученная на истории тиков того же актива.

ТОЛЬКО ДЕМО: счёт выбирается по номеру (DERIV_ACCOUNT_ID), адрес WebSocket должен содержать /ws/demo, иначе бот останавливается.
Режима реальных денег в нём нет сознательно. Токен и номер счёта хранятся только в .env (DERIV_TOKEN, DERIV_APP_ID, DERIV_ACCOUNT_ID).

Как работает:
  1. Скачивает историю тиков символа (публичная лента, без ключа), обучает градиентный бустинг предсказывать, будет ли цена через N тиков
     выше, чем на следующем тике (вход контракта — следующий тик после покупки). Порог уверенности — топ-20% по обучению.
  2. Подписывается на живые тики; после каждого тика считает признаки; если уверенность выше порога и нет открытого контракта —
     покупает CALL/PUT на N тиков фиксированной ставкой (демо).
  3. Каждую ставку записывает в deriv_log.csv, показывает РЕАЛЬНУЮ выплату из ответа Deriv и честную статистику с интервалом.
     Ничья считается проигрышем (так при выключенных «равных значениях»), как и в вашей автоматизации.

  pip install scikit-learn websocket-client requests
  python deriv_tick_bot.py --payout-check                 # только спросить у Deriv выплату по символам (ничего не покупает)
  python deriv_tick_bot.py --symbol cryBTCUSD --stake 1 --total 1000
  python deriv_tick_bot.py --selftest                     # проверка логики на имитации

Не могу проверить на настоящем Deriv из своего окружения. Если API ответит ошибкой или формат окажется другим, включите DERIV_DEBUG=1
(печатает сырые ответы) и пришлите текст ошибки без токена.
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

import config  # noqa: F401  (подгружает .env)
import deriv_demo_bot as B
import deriv_ticks_test as T

PUBLIC_WS = os.getenv("DERIV_WS_URL", "wss://api.derivws.com/trading/v1/options/ws/public")
DEBUG = os.getenv("DERIV_DEBUG", "") == "1"
LAGS = 20


def train_model(prices, ahead, top, log=print):
    """Обучает модель на истории тиков. Возвращает (модель, порог уверенности, масштаб волатильности)."""
    X, y, tie = T.make_xy(prices, ahead)
    vol = float(np.std(np.diff(np.log(prices))) + 1e-12)
    model = T.HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=5.0,
                                             min_samples_leaf=200, early_stopping=False, random_state=0).fit(X, y)
    thr = float(np.quantile(np.abs(model.predict_proba(X)[:, 1] - 0.5), 1 - top))
    log(f"Модель обучена на {len(y)} примерах (доля ничьих {100 * tie:.1f}%); порог уверенности {thr:.3f}, топ-{100 * top:.0f}% сигналов.")
    return model, thr, vol


def live_features(prices, vol):
    """Те же признаки, что в make_xy, для самой последней цены (последние LAGS изменений + суммы + разброс за 60 тиков)."""
    r = np.diff(np.log(np.asarray(prices, dtype=float)))
    if len(r) < LAGS + 60:
        return None
    w = r[-LAGS:] / vol
    return np.array(list(w) + [w[-5:].sum(), w[-10:].sum(), w.sum(), r[-60:].std() / vol])


def signal(model, thr, prices, vol):
    f = live_features(prices, vol)
    if f is None:
        return None
    p = float(model.predict_proba(f.reshape(1, -1))[0, 1])
    if abs(p - 0.5) < thr:
        return None
    return (1 if p > 0.5 else -1), p


def tick_stream(symbol, ws=None, log=print):
    """Генератор живых тиков (цена, время). Подписка на публичную ленту; при обрыве переподключается."""
    import websocket
    own = ws is None
    keys = ["ticks", "underlying_symbol"]                  # новый API мог переименовать поле; пробуем по очереди
    attempt = 0
    got_any = False
    while True:
        try:
            if own:
                ws = websocket.create_connection(PUBLIC_WS, timeout=20)
            req = {"ticks": symbol, "subscribe": 1} if keys[attempt % 2] == "ticks" else {"ticks": symbol, "underlying_symbol": symbol, "subscribe": 1}
            ws.send(json.dumps(req))
            last_ping = time.time()
            while True:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    ws.send(json.dumps({"ping": 1}))
                    continue
                msg = json.loads(raw)
                if DEBUG:
                    print("DEBUG tick msg:", str(msg)[:200])
                if msg.get("error"):
                    raise RuntimeError(f"Deriv API (ticks): {msg['error'].get('message')}")
                t = msg.get("tick")
                if t and "quote" in t:
                    got_any = True
                    yield float(t["quote"]), t.get("epoch")
                if time.time() - last_ping > 20:
                    ws.send(json.dumps({"ping": 1}))
                    last_ping = time.time()
        except (websocket.WebSocketException, OSError, RuntimeError) as e:
            if not own:
                raise
            attempt += 1
            log(f"  лента тиков оборвалась ({type(e).__name__}: {str(e)[:80]}), переподключаюсь (попытка {attempt})...")
            if attempt >= 8 and not got_any:
                raise SystemExit("Лента тиков не отдаёт данные. Запустите с DERIV_DEBUG=1 (cmd: set DERIV_DEBUG=1) и пришлите вывод.")
            time.sleep(min(10, 2 * attempt))


def payout_check(symbols, expiry, unit):
    """Спрашивает у Deriv цену контракта (proposal) и печатает выплату в процентах. Ничего не покупает."""
    trader = B.Trader(B.connect(), 1.0, expiry, 10 ** 9, 10 ** 9, unit=unit)
    print(f"Выплата по Rise/Fall на {expiry}{'т' if unit == 't' else ' мин'} (ставка 1 {trader.currency}); ничего не покупается:\n")
    print(f"  {'символ':<14}{'CALL (вверх)':>16}{'PUT (вниз)':>16}   порог безубыточности")
    errs = {}
    for sym in symbols:
        cells = []
        for ct in ("CALL", "PUT"):
            try:
                pr = trader.c.call({"proposal": 1, "amount": 1, "basis": "stake", "contract_type": ct, "currency": trader.currency,
                                    "duration": expiry, "duration_unit": unit, "underlying_symbol": sym})["proposal"]
                pay = float(pr["payout"]) / float(pr["ask_price"]) - 1
                cells.append(f"{100 * pay:.1f}%")
            except Exception as e:  # noqa: BLE001
                cells.append("—")
                errs.setdefault(sym, str(e)[:300])
        be = ""
        try:
            p0 = float(cells[0].rstrip("%")) / 100
            be = f"{100 / (1 + p0):.2f}%"
        except ValueError:
            pass
        print(f"  {sym:<14}{cells[0]:>16}{cells[1]:>16}   {be}")
    print("\n«—» значит, что Deriv не принял запрос. Ответ Deriv:")
    for sym, e in errs.items():
        print(f"  {sym}: {e}")
    if errs:
        sym = next(iter(errs))
        try:
            try:
                cf = trader.c.call({"contracts_for": sym})
            except Exception:  # noqa: BLE001
                cf = trader.c.call({"contracts_for": 1, "underlying_symbol": sym})
            av = cf.get("contracts_for", cf).get("available", [])
            seen = sorted({(c.get("contract_type"), c.get("min_contract_duration"), c.get("max_contract_duration")) for c in av})
            print(f"\nДоступные контракты для {sym} (тип, мин. срок, макс. срок), первые 15:")
            for row in seen[:15]:
                print("  ", row)
        except Exception as e:  # noqa: BLE001
            print(f"contracts_for для {sym} тоже не сработал: {str(e)[:200]}")


def run_live(trader, symbol, model, thr, vol, ahead, total, ticks, log=print, report_every=25, max_ticks=None):
    """Основной цикл: тик -> (проверить открытый контракт) -> сигнал -> ставка."""
    buf, placed, n = [], 0, 0
    for price, epoch in ticks:
        n += 1
        buf.append(price)
        buf = buf[-200:]
        if trader.open:
            trader.poll()
        if trader.halted and not trader.open:
            break
        if not trader.open and not trader.halted and placed < total:
            sig = signal(model, thr, buf, vol)
            if sig:
                d, p = sig
                if trader.place(symbol, d, price, symbol=symbol):
                    placed += 1
                    log(f"  сигнал: {'ВВЕРХ' if d > 0 else 'ВНИЗ'} (уверенность {max(p, 1 - p):.2f}), цена {price}")
                    if placed % report_every == 0:
                        log(f"[{datetime.now():%H:%M}] {trader.report()}")
        if placed >= total and not trader.open:
            break
        if max_ticks and n >= max_ticks:
            break
    for _ in range(60):                                               # дождаться последнего контракта
        if not trader.open:
            break
        trader.poll()
        time.sleep(1)
    log("ИТОГ: " + trader.report())


# ---------------- самопроверка ----------------
def selftest():
    import tempfile
    from deriv_demo_bot import FakeClient
    B.LOG = os.path.join(tempfile.mkdtemp(), "t.csv")
    ok = []
    # 1) признаки для живых тиков совпадают с признаками обучения
    rng = np.random.default_rng(5)
    prices = np.round(1000 * np.exp(np.cumsum(rng.normal(0, 0.0005, 3000))), 2)
    X, y, tie = T.make_xy(prices, 5)
    vol = float(np.std(np.diff(np.log(prices))) + 1e-12)
    t_idx = 500
    f_live = live_features(prices[:t_idx + 1], vol)
    r_all = np.diff(np.log(prices))
    w = r_all[t_idx - LAGS:t_idx] / vol
    f_train = np.array(list(w) + [w[-5:].sum(), w[-10:].sum(), w.sum(), r_all[t_idx - 60:t_idx].std() / vol])
    ok.append(("живые признаки совпадают с признаками обучения", np.allclose(f_live, f_train)))
    # 2) цикл на искусственном потоке с «импульсом»: бот делает ставки, журнал заполняется, лимит соблюдается
    mom = [1000.0]
    r = 0.0
    for _ in range(20000):
        r = 0.6 * r + rng.normal(0, 0.0004)
        mom.append(round(mom[-1] * np.exp(r), 2))
    mom = np.array(mom)
    model, thr, vol2 = train_model(mom[:15000], 5, 0.2, log=lambda *a: None)
    outcomes = ["won", "lost"] * 200
    tr = B.Trader(FakeClient(outcomes=outcomes), 1, 5, 10 ** 9, 10 ** 6, log=lambda *a: None, unit="t")
    stream = ((float(p), i) for i, p in enumerate(mom[15000:]))
    run_live(tr, "cryBTCUSD", model, thr, vol2, 5, 30, stream, log=lambda *a: None)
    ok.append(("бот делает ровно заданное число ставок и закрывает их", len(tr.done) == 30 and not tr.open))
    ok.append(("отчёт содержит интервал и порог", "интервал" in tr.report() and "порог" in tr.report()))
    # 3) бот останавливается по лимиту потерь
    tr2 = B.Trader(FakeClient(outcomes=["lost"] * 100), 1, 5, 10 ** 9, 3, log=lambda *a: None, unit="t")
    run_live(tr2, "cryBTCUSD", model, thr, vol2, 5, 1000, ((float(p), i) for i, p in enumerate(mom[15000:])), log=lambda *a: None)
    ok.append(("остановка по лимиту потерь", tr2.halted and len(tr2.done) <= 5))
    bad = 0
    for name, good in ok:
        print(("  ok   " if good else "  ОШИБКА  ") + name)
        bad += not good
    print(f"\nИтог: {len(ok) - bad}/{len(ok)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="cryBTCUSD")
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--ahead", type=int, default=5, help="срок контракта в тиках")
    ap.add_argument("--top", type=float, default=0.2, help="доля самых уверенных сигналов")
    ap.add_argument("--total", type=int, default=1000, help="сколько ставок сделать")
    ap.add_argument("--ticks-history", type=int, default=100000)
    ap.add_argument("--max-loss", type=float, default=20.0)
    ap.add_argument("--payout-check", action="store_true", help="только спросить у Deriv выплату (ничего не покупать)")
    ap.add_argument("--symbols", default="cryBTCUSD,cryETHUSD,R_100", help="для --payout-check")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if not B.TOKEN:
        raise SystemExit("Впишите DERIV_TOKEN, DERIV_APP_ID и DERIV_ACCOUNT_ID (номер ДЕМО-счёта) в .env. Токен никому не присылайте.")
    if not B.ACCOUNT_ID:
        raise SystemExit("Нужен DERIV_ACCOUNT_ID — номер ДЕМО-счёта (бот работает только на демо через OTP-адрес /ws/demo).")
    if a.payout_check:
        payout_check([s.strip() for s in a.symbols.split(",") if s.strip()], a.ahead, "t")
        return
    prices = T.fetch_ticks(a.symbol, a.ticks_history)
    model, thr, vol = train_model(prices, a.ahead, a.top)
    trader = B.Trader(B.connect(), a.stake, a.ahead, 10 ** 9, a.max_loss, unit="t")
    print(f"Работает на ДЕМО: {a.symbol}, {a.ahead} тиков, ставка {a.stake}, всего {a.total} ставок, лимит потерь {a.max_loss}. Остановка: Ctrl+C.")
    try:
        run_live(trader, a.symbol, model, thr, vol, a.ahead, a.total, tick_stream(a.symbol))
    except KeyboardInterrupt:
        print("\nОстановлено. " + trader.report())


if __name__ == "__main__":
    main()
