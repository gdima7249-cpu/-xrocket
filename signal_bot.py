"""Бот-подсказчик «вверх/вниз» для ДЕМО-счёта (вы открываете сделки сами, бот только говорит направление) + честный журнал результатов.

Модель (градиентный бустинг, как в binary_research.py) учится на последних днях минутных свечей Binance и каждую минуту смотрит на
реальные цены. Подаёт сигнал только когда уверена сильнее порога (порог выбирается по обучению: топ-20% самых уверенных).
Каждый сигнал бот САМ проверяет через срок ставки по реальным ценам Binance и считает долю угаданных — в том числе со сдвигом
входа на минуту (имитация вашей задержки, пока вы открываете сделку руками) — и сравнивает с порогом безубыточности для выплаты.

ВАЖНО: на Pocket Option цены OTC-активов рисует сама площадка, они не совпадают с Binance. Подсказки считаются по настоящим ценам
крипты Binance, поэтому используйте их только на ДЕМО-счёте и только для настоящих (не OTC) криптоактивов, чтобы сравнивать
свои результаты с журналом бота. Реальными деньгами по этим сигналам не торгуйте, пока журнал не покажет устойчивый результат
выше порога (обычно нужно 500–1000 сигналов).

  pip install scikit-learn
  python signal_bot.py --interactive    # пишете монету (btc, eth, sol...) — бот сразу отвечает направлением
  python signal_bot.py                  # обучение (несколько минут при первом запуске), затем сигналы каждую минуту
  python signal_bot.py --expiry 5 --payout 0.92 --top 0.2 --pairs BTC,ETH,SOL
  python signal_bot.py --selftest       # проверка логики без интернета
Журнал: signals_log.csv (время, монета, направление, уверенность, цена, итог). Остановка: Ctrl+C.
"""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from binary_research import FEATURES, PAIRS, download, features, make_gbm

KLINES = "https://data-api.binance.vision/api/v3/klines"
LOG = "signals_log.csv"


def fetch_1m(base: str, limit: int = 300) -> pd.DataFrame:
    """Последние закрытые минутные свечи (с потоком покупок taker)."""
    r = requests.get(KLINES, params={"symbol": base + "USDT", "interval": "1m", "limit": limit}, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
    df.columns = ["time", "open", "high", "low", "close", "volume", "taker_buy"]
    df = df.astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.iloc[:-1].reset_index(drop=True)                      # последняя свеча ещё не закрыта


def train(pairs, days, horizon, top, loader=download, log=print):
    frames = {s: loader(s + "USDT", days) for s in pairs}
    data = pd.concat([features(df, horizon).assign(pair=s) for s, df in frames.items()])
    X, y = data[FEATURES].to_numpy(), data["y"].to_numpy()
    model = make_gbm().fit(X, y)
    thr = float(np.quantile(np.abs(model.predict_proba(X)[:, 1] - 0.5), 1 - top))
    log(f"Обучено на {len(data)} примерах ({len(pairs)} монет, {days} дн.); порог уверенности {thr:.3f} (|p-0.5|), топ-{100 * top:.0f}% сигналов.")
    return model, thr


def signal_for(model, thr, df, horizon):
    """(направление +1/-1, уверенность p, цена закрытия) по последней закрытой свече, либо None."""
    f = features(df, horizon, live=True)
    if f.empty or f["time"].iloc[-1] != df["time"].iloc[-1]:
        return None
    p = float(model.predict_proba(f[FEATURES].iloc[[-1]].to_numpy())[0, 1])
    if abs(p - 0.5) < thr:
        return None
    return (1 if p > 0.5 else -1), p, float(df["close"].iloc[-1])


def outcome(direction, entry, exit_):
    """+1 выигрыш, -1 проигрыш, 0 ничья (цена не изменилась — ставка возвращается)."""
    if exit_ == entry:
        return 0
    return 1 if (exit_ > entry) == (direction > 0) else -1


def report(rows, payout, expiry):
    be = 1 / (1 + payout)
    done = [r for r in rows if r.get("res") is not None and r["res"] != 0]
    done_d = [r for r in rows if r.get("res_d") is not None and r["res_d"] != 0]
    if not done:
        return "Пока нет проверенных сигналов."
    w = np.mean([r["res"] > 0 for r in done])
    line = (f"Сигналов проверено {len(done)}: угадано {100 * w:.1f}% (порог при выплате {100 * payout:.0f}% — {100 * be:.2f}%), "
            f"ROI на ставку {100 * (w * payout - (1 - w)):+.1f}%")
    if done_d:
        wd = np.mean([r["res_d"] > 0 for r in done_d])
        line += f";  с задержкой входа в минуту: {100 * wd:.1f}% (ROI {100 * (wd * payout - (1 - wd)):+.1f}%)"
    if len(done) < 300:
        line += "  [мало сигналов, <300: выводов делать нельзя]"
    return line


def beep():
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:  # noqa: BLE001
        print("\a", end="")


def log_row(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["время", "монета", "направление", "уверенность", "цена", "итог", "итог_с_задержкой", "сильный"])
        w.writerow([row["t"].strftime("%Y-%m-%d %H:%M:%S"), row["pair"], "ВВЕРХ" if row["dir"] > 0 else "ВНИЗ",
                    f"{row['p']:.3f}", row["price"], row.get("res"), row.get("res_d"), int(row.get("strong", True))])


def resolve(pending, rows, frames, expiry):
    """Проверяет сигналы, у которых прошёл срок (и ещё минута для варианта с задержкой), по реальным ценам Binance."""
    for r in pending[:]:
        df = frames.get(r["pair"])
        if df is None:
            continue
        t_exit = r["t"] + pd.Timedelta(minutes=expiry)
        t_exit_d = t_exit + pd.Timedelta(minutes=1)
        if df["time"].iloc[-1] < t_exit_d:
            continue

        def px(tt):
            m = df["time"] == tt
            return float(df.loc[m, "close"].iloc[0]) if m.any() else None

        x, e_d, x_d = px(t_exit), px(r["t"] + pd.Timedelta(minutes=1)), px(t_exit_d)
        if None not in (x, e_d, x_d):
            r["res"], r["res_d"] = outcome(r["dir"], r["price"], x), outcome(r["dir"], e_d, x_d)
            pending.remove(r)
            log_row(r)
            rows.append(r)


def run(a, fetch=fetch_1m, loader=download, sleep=time.sleep, now=time.time, max_iter=None, log=print):
    pairs = a.pairs.split(",")
    model, thr = train(pairs, a.days, a.expiry, a.top, loader, log)
    trained_at = now()
    pending, rows, last_min, last_report, it = [], [], 0, now(), 0
    log("Готов. Ждите сигналы (каждую минуту проверка). Открывайте сделку на ДЕМО-счёте на %d мин. Остановка: Ctrl+C." % a.expiry)
    while max_iter is None or it < max_iter:
        it += 1
        cur_min = int(now() // 60)
        if cur_min != last_min:
            last_min = cur_min
            if now() - trained_at > 24 * 3600:                      # переобучение раз в сутки на свежих данных
                model, thr = train(pairs, a.days, a.expiry, a.top, loader, log)
                trained_at = now()
            frames = {}
            for s in pairs:
                try:
                    frames[s] = fetch(s)
                except Exception as e:  # noqa: BLE001
                    log(f"  {s}: {type(e).__name__}: {str(e)[:80]}")
            resolve(pending, rows, frames, a.expiry)
            # новые сигналы
            for s, df in frames.items():
                if any(p["pair"] == s for p in pending):              # не повторяем сигнал по той же монете, пока идёт прошлый
                    continue
                sig = signal_for(model, thr, df, a.expiry)
                if sig:
                    d, p, price = sig
                    r = {"t": df["time"].iloc[-1], "pair": s, "dir": d, "p": p, "price": price, "strong": True}
                    pending.append(r)
                    log(f"\n>>> СИГНАЛ {datetime.now():%H:%M:%S}  {s}/USDT  {'ВВЕРХ ▲' if d > 0 else 'ВНИЗ ▼'}  "
                        f"(уверенность {max(p, 1 - p):.2f}, цена {price:.6g}, срок {a.expiry} мин)")
                    beep()
            if now() - last_report >= 600 and rows:
                last_report = now()
                log(f"[{datetime.now():%H:%M}] {report(rows, a.payout, a.expiry)}")
        sleep(5)
    return rows


def probability(model, df, horizon):
    """(p вверх, цена закрытия, время свечи) по последней закрытой свече — без порога уверенности."""
    f = features(df, horizon, live=True)
    if f.empty or f["time"].iloc[-1] != df["time"].iloc[-1]:
        return None
    return float(model.predict_proba(f[FEATURES].iloc[[-1]].to_numpy())[0, 1]), float(df["close"].iloc[-1]), df["time"].iloc[-1]


def interactive(a, fetch=fetch_1m, loader=download, input_fn=input, log=print, background=True):
    """Вы пишете монету (btc, eth, sol, любую с Binance) — бот сразу отвечает направлением. Результаты проверяются сами."""
    import threading
    pairs = a.pairs.split(",")
    model, thr = train(pairs, a.days, a.expiry, a.top, loader, log)
    pending, rows, lock = [], [], threading.Lock()
    stop = threading.Event()

    def worker():
        while not stop.wait(30):
            with lock:
                frames = {}
                for base in {r["pair"] for r in pending}:
                    try:
                        frames[base] = fetch(base)
                    except Exception:  # noqa: BLE001
                        pass
                resolve(pending, rows, frames, a.expiry)

    if background:
        threading.Thread(target=worker, daemon=True).start()
    log("Команды: название монеты (btc, eth, sol, doge, любая с Binance) — сигнал сразу; stat — итоги; help; q — выход.")
    while True:
        try:
            cmd = input_fn("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit", "выход"):
            break
        if cmd in ("", "help", "?"):
            log("Пример: btc   -> сигнал по BTC/USDT.   stat -> сколько сигналов угадано.   q -> выход.")
            continue
        if cmd in ("stat", "стат", "итоги"):
            with lock:
                done = list(rows)
            strong = [r for r in done if r["strong"]]
            weak = [r for r in done if not r["strong"]]
            log(f"Все: {report(done, a.payout, a.expiry)}")
            if strong:
                log(f"Только СИЛЬНЫЕ: {report(strong, a.payout, a.expiry)}")
            if weak:
                log(f"Только слабые: {report(weak, a.payout, a.expiry)}")
            log(f"Ждут проверки: {len(pending)}")
            continue
        base = cmd.replace("/usdt", "").replace("usdt", "").upper()
        try:
            df = fetch(base)
        except Exception as e:  # noqa: BLE001
            log(f"  Не удалось получить цены {base}: {type(e).__name__} (проверьте название монеты)")
            continue
        res = probability(model, df, a.expiry)
        if not res:
            log("  Недостаточно данных для расчёта.")
            continue
        p, price, t = res
        strong = abs(p - 0.5) >= thr
        d = 1 if p > 0.5 else -1
        age = max(0, int(datetime.now().timestamp() - (t.timestamp() + 60)))
        log(f"  {base}/USDT  {'ВВЕРХ ▲' if d > 0 else 'ВНИЗ ▼'}  уверенность {max(p, 1 - p):.2f}  "
            f"[{'СИЛЬНЫЙ сигнал' if strong else 'слабый: ниже порога, это почти монетка'}]  цена {price:.6g}, срок {a.expiry} мин "
            f"(данные свечи {age} с назад)")
        beep()
        with lock:
            pending.append({"t": t, "pair": base, "dir": d, "p": p, "price": price, "strong": strong})
    stop.set()
    log("Выход.")
    return rows


def selftest():
    ok = []
    ok.append(("выигрыш: ставка вверх, цена выросла", outcome(1, 100, 101) == 1))
    ok.append(("проигрыш: ставка вверх, цена упала", outcome(1, 100, 99) == -1))
    ok.append(("ставка вниз, цена упала — выигрыш", outcome(-1, 100, 99) == 1))
    ok.append(("ничья — возврат ставки", outcome(1, 100, 100) == 0))
    rows = [{"res": 1, "res_d": 1}] * 6 + [{"res": -1, "res_d": -1}] * 4
    txt = report(rows, 0.92, 5)
    ok.append(("отчёт: 60% угадано, порог 52.08%, ROI +15.2%", "60.0%" in txt and "52.08%" in txt and "+15.2%" in txt))
    # проверка сигнала на искусственном ряде с вшитой закономерностью (поток покупок прошлой минуты)
    from binary_research import synthetic
    pairs = ["BTC", "ETH"]
    frames = {s: synthetic(20 * 1440, i, True) for i, s in enumerate(pairs)}
    model, thr = train(pairs, 20, 1, 0.2, loader=lambda sym, d: frames[sym[:-4]], log=lambda *a: None)
    df = frames["BTC"].iloc[:-1000].reset_index(drop=True)
    wins = n = 0
    for k in range(200, 1000, 3):                                    # много точек подряд: направление сигнала должно совпадать с исходом чаще 50%
        sub = frames["BTC"].iloc[:len(df) + k].reset_index(drop=True)
        sig = signal_for(model, thr, sub.iloc[:-1].reset_index(drop=True), 1)
        if sig:
            res = outcome(sig[0], sub["close"].iloc[-2], sub["close"].iloc[-1])
            if res:
                n += 1
                wins += res > 0
    ok.append((f"на данных с закономерностью сигналы угадывают заметно чаще 50% ({wins}/{n})", n >= 30 and wins / n > 0.6))
    bad = 0
    for name, good in ok:
        print(("  ok   " if good else "  ОШИБКА  ") + name)
        bad += not good
    print(f"\nИтог: {len(ok) - bad}/{len(ok)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--days", type=int, default=30, help="сколько дней истории для обучения")
    ap.add_argument("--expiry", type=int, default=5, help="срок ставки в минутах")
    ap.add_argument("--top", type=float, default=0.2, help="доля самых уверенных сигналов")
    ap.add_argument("--payout", type=float, default=0.92)
    ap.add_argument("--interactive", action="store_true", help="режим «команда -> сигнал»: пишете монету, бот сразу отвечает")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    try:
        (interactive if a.interactive else run)(a)
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
