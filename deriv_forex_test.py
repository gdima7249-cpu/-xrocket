"""Можно ли угадывать направление РЕАЛЬНЫХ валютных пар Deriv (frxGBPCAD, frxEURUSD ...) на 15/30/60 минут лучше порога выплаты?

Ставки Rise/Fall на валюты Deriv даёт от 15 минут. Скрипт:
  1) качает 5-минутные свечи пары из публичной ленты Deriv (ключ не нужен);
  2) walk-forward (учимся на прошлом, проверяем на будущем): модель по последним свечам и правило отката
     «резкое падение (в единицах обычного разброса) → вверх»; доля угаданных против порога безубыточности;
  3) с флагом --payout спрашивает у Deriv (демо-счёт из .env) реальную выплату на этих сроках. Ничего не покупает.

  python deriv_forex_test.py --symbol frxGBPCAD
  python deriv_forex_test.py --symbol frxEURUSD --payout
  python deriv_forex_test.py --symbol frxEURUSD,frxGBPUSD,frxUSDJPY,frxAUDUSD,frxUSDCAD --payout   # несколько пар + общий итог
  python deriv_forex_test.py --symbol frxGBPNZD,frxEURNZD,frxGBPAUD --delay 1   # вход на 5 минут позже сигнала
  python deriv_forex_test.py --list                  # какие символы есть на Deriv (индексы, сырьё, валюты)
  python deriv_forex_test.py --synthetic              # случайные цены: везде должно быть около 50%
"""
import argparse
import json
import time

import numpy as np

import deriv_ticks_test as T

LAGS = 12


def fetch_candles(symbol, total, gran=300, ws=None, pause=2.0):
    """5-минутные свечи (по 5000 за запрос, назад во времени), без дублей, по возрастанию времени: (время, close).
    Deriv ограничивает частоту таких запросов: между запросами пауза, при упоре в лимит — ожидание и повтор.
    Скачанное сохраняется в research_cache (при повторном запуске в течение суток берётся оттуда)."""
    import os
    os.makedirs("research_cache", exist_ok=True)
    path = f"research_cache/DERIV_{symbol}_{gran}s_{total}.json"
    if ws is None and os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        print(f"Свечи {symbol} взяты из кэша: {len(d['t'])}")
        return np.array(d["t"]), np.array(d["c"])
    own = ws is None
    if own:
        import websocket
        ws = websocket.create_connection(T.WS_URL, timeout=30)
    seen, end, req, stalled, waits = {}, "latest", 0, 0, 0
    try:
        while len(seen) < total and req < 200 and stalled < 3:
            ws.send(json.dumps({"ticks_history": symbol, "style": "candles", "granularity": gran, "count": 5000,
                                "end": end, "adjust_start_time": 1}))
            req += 1
            msg = json.loads(ws.recv())
            if msg.get("error"):
                text = str(msg["error"].get("message", ""))
                if "rate limit" in text.lower() and waits < 10:
                    waits += 1
                    wait = 30 * waits
                    print(f"  Deriv просит подождать (лимит запросов), жду {wait} с... (уже скачано {len(seen)} свечей)")
                    time.sleep(wait)
                    continue
                raise SystemExit(f"Deriv API: {text} (проверьте символ, например frxGBPCAD)")
            cs = msg.get("candles") or []
            if not cs:
                break
            before = len(seen)
            for c in cs:
                seen.setdefault(int(c["epoch"]), float(c["close"]))
            stalled = stalled + 1 if len(seen) - before < len(cs) * 0.5 else 0
            end = min(int(c["epoch"]) for c in cs) - 1
            print(f"  скачано {len(seen)} свечей...")
            time.sleep(pause)
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass
    ts = sorted(seen)
    if not ts:
        raise SystemExit("Свечей нет")
    print(f"Свечей: {len(ts)} (запросов {req}); период {time.strftime('%Y-%m-%d', time.gmtime(ts[0]))} … "
          f"{time.strftime('%Y-%m-%d', time.gmtime(ts[-1]))}")
    if own:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"t": ts, "c": [seen[t] for t in ts]}, f)
    return np.array(ts), np.array([seen[t] for t in ts])


DELAY = 0


def features(c, h, ts=None, gran=300):
    """X — последние LAGS изменений (в долях обычного разброса) и их суммы; y — цена через h свечей выше текущей (1) или ниже (0).
    Если переданы времена свечей ts, окна, где между сигналом и выходом есть перерыв в торговле (ночь, выходные, праздники),
    отбрасываются: иначе разрыв цены на открытии рынка выглядел бы как «резкое движение за 15 минут»."""
    r = np.diff(np.log(c))
    vol = np.std(r) + 1e-12
    X, Y, Z = [], [], []
    d = DELAY
    gaps = None
    if ts is not None:
        gaps = np.r_[0, np.cumsum(np.diff(np.asarray(ts)) != gran)]   # число перерывов до каждой свечи
    for t in range(LAGS + 48, len(r) - h - d):
        if gaps is not None and gaps[t + d + h] != gaps[t - 3]:         # окно сигнал→выход пересекает перерыв
            continue
        w = r[t - LAGS:t] / vol
        fut = c[t + d + h] - c[t + d]                               # вход через d свечей после сигнала
        if fut == 0:
            continue
        X.append(list(w) + [w[-3:].sum(), w[-6:].sum(), w.sum(), r[t - 48:t].std() / vol])
        Y.append(1.0 if fut > 0 else 0.0)
        Z.append(w[-3:].sum())                                      # изменение за последние 15 минут (в разбросах)
    return np.array(X), np.array(Y), np.array(Z)


def walk(X, Y, Z, h, folds=5):
    n = len(Y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    res = {"модель, все": [], "модель, уверенные 20%": [], "откат: падение >3 разбросов за 15м → вверх": [],
           "откат: рост >3 разбросов за 15м → вниз": [], "всегда вверх": [],
           "откат, сила падения 3–4 → вверх": [], "откат, сила падения 4–5 → вверх": [], "откат, сила падения >5 → вверх": []}
    for k in range(1, folds):
        tr, te = slice(0, edges[k] - h), slice(edges[k], edges[k + 1])
        m = T.HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=5.0,
                                             min_samples_leaf=200, early_stopping=False, random_state=0).fit(X[tr], Y[tr])
        p = m.predict_proba(X[te])[:, 1]
        thr = np.quantile(np.abs(m.predict_proba(X[tr])[:, 1] - 0.5), 0.8)
        y = Y[te]
        side = (p > 0.5).astype(float)
        conf = np.abs(p - 0.5) >= thr
        res["модель, все"].append(((side == y).sum(), len(y)))
        res["модель, уверенные 20%"].append(((side[conf] == y[conf]).sum(), int(conf.sum())))
        dip = Z[te] <= -3 * np.sqrt(3)                              # падение больше 3 «обычных разбросов» за 15 мин
        res["откат: падение >3 разбросов за 15м → вверх"].append((int(y[dip].sum()), int(dip.sum())))
        pump = Z[te] >= 3 * np.sqrt(3)                              # рост больше 3 разбросов за 15 мин → ставка вниз
        res["откат: рост >3 разбросов за 15м → вниз"].append((int((1 - y[pump]).sum()), int(pump.sum())))
        res["всегда вверх"].append((int(y.sum()), len(y)))
        zz = -Z[te] / np.sqrt(3)                                    # сила падения в тех же единицах, что печатает бот («−4.7 разброса»)
        for lo, hi, name in ((3, 4, "3–4"), (4, 5, "4–5"), (5, 1e9, ">5")):
            m = (zz >= lo) & (zz < hi)
            res[f"откат, сила падения {name} → вверх"].append((int(y[m].sum()), int(m.sum())))
    return res


def payouts(symbol, minutes):
    import deriv_demo_bot as B
    tr = B.Trader(B.connect(), 1.0, 1, 10 ** 9, 10 ** 9, unit="m")
    out = {}
    for mnt in minutes:
        try:
            pr = tr.c.call({"proposal": 1, "amount": 10, "basis": "stake", "contract_type": "CALL", "currency": tr.currency,
                            "duration": mnt, "duration_unit": "m", "underlying_symbol": symbol})["proposal"]
            out[mnt] = float(pr["payout"]) / float(pr["ask_price"]) - 1
        except Exception as e:  # noqa: BLE001
            print(f"  выплата на {mnt} мин: {str(e)[:120]}")
    return out


POOL = []                                                            # события «резкое падение» по всем парам (для модели уверенности)


def pool_events(symbol, c, ts, h=6, gran=300):
    """События «падение >3 разбросов за 15 мин» для срока h свечей (по умолчанию 30 мин) с признаками для модели уверенности."""
    r = np.diff(np.log(c))
    vol = np.std(r) + 1e-12
    d = DELAY
    gaps = np.r_[0, np.cumsum(np.diff(np.asarray(ts)) != gran)] if ts is not None else None
    cross = 0.0 if "USD" in symbol else 1.0
    for t in range(LAGS + 48, len(r) - h - d):
        z = r[t - 3:t].sum() / vol / np.sqrt(3)
        if z > -3:
            continue
        if gaps is not None and gaps[t + d + h] != gaps[t - 3]:
            continue
        fut = c[t + d + h] - c[t + d]
        if fut == 0:
            continue
        hour = (ts[t] % 86400) / 3600 if ts is not None else 0.0
        f = [-z, r[t - 48:t].std() / vol, r[t - 12:t - 3].sum() / vol, r[t - 48:t - 12].sum() / vol,
             np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), cross]
        POOL.append((float(ts[t]) if ts is not None else float(t), f, 1.0 if fut > 0 else 0.0))


def confidence_report(folds=4):
    """Учимся на прошлых сигналах (все пары вместе, по времени), предсказываем будущие; делим на трети по уверенности."""
    if len(POOL) < 400:
        print(f"\nМодель уверенности: мало сигналов ({len(POOL)}), нужно хотя бы 400.")
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ev = sorted(POOL, key=lambda e: e[0])
    X = np.array([e[1] for e in ev])
    Y = np.array([e[2] for e in ev])
    edges = np.linspace(0, len(Y), folds + 1).astype(int)
    P, YY = [], []
    for k in range(1, folds):
        tr, te = slice(0, edges[k]), slice(edges[k], edges[k + 1])
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.3, max_iter=1000).fit(sc.transform(X[tr]), Y[tr])
        P.append(m.predict_proba(sc.transform(X[te]))[:, 1])
        YY.append(Y[te])
    P, YY = np.concatenate(P), np.concatenate(YY)
    q1, q2 = np.quantile(P, [1 / 3, 2 / 3])
    print(f"\nМОДЕЛЬ УВЕРЕННОСТИ (срок 30 мин; учится на прошлых сигналах всех пар, проверка на будущих; {len(YY)} сигналов):")
    for name, msk in (("низкая уверенность", P < q1), ("средняя", (P >= q1) & (P < q2)), ("высокая уверенность", P >= q2)):
        n = int(msk.sum())
        p = YY[msk].mean() if n else float("nan")
        ci = 1.96 * np.sqrt(p * (1 - p) / n) if n else float("nan")
        print(f"  {name:<22}{n:>6} сигналов, угадано {100 * p:5.1f}% ± {100 * ci:.1f}")
    print("  Если «высокая» заметно выше «низкой» (разница больше погрешностей), ставить больше на уверенные сигналы имеет смысл.")


def list_symbols():
    """Печатает символы публичной ленты Deriv по рынкам (кроме синтетических индексов, у которых цену рисует генератор)."""
    import websocket
    ws = websocket.create_connection(T.WS_URL, timeout=30)
    ws.send(json.dumps({"active_symbols": "brief"}))
    msg = json.loads(ws.recv())
    ws.close()
    if msg.get("error"):
        raise SystemExit(f"Deriv API: {msg['error'].get('message')}")
    items = msg.get("active_symbols") or []
    if not items:
        raise SystemExit(f"Пустой ответ: {str(msg)[:300]}")
    groups = {}
    for it in items:
        sym = it.get("underlying_symbol") or it.get("symbol") or "?"
        market = it.get("market_display_name") or it.get("market") or "?"
        name = it.get("display_name") or it.get("underlying_symbol_name") or ""
        if "synthetic" in str(market).lower() or "derived" in str(market).lower():
            continue
        groups.setdefault(market, []).append(f"{sym:<14}{name}")
    for market, rows in sorted(groups.items()):
        print(f"\n{market}:")
        for r in sorted(rows):
            print("  " + r)
    print("\nДля проверки подставьте символы через запятую:  python deriv_forex_test.py --symbol СИМВОЛ1,СИМВОЛ2 --payout")


def run_one(symbol, candles, payout, synthetic, total):
    """Проверка одной пары; добавляет результаты правила отката в total[срок] = [угадано, ставок, выплата]."""
    if synthetic:
        c, ts = T.synthetic(candles), None
    else:
        ts, c = fetch_candles(symbol, candles)
    if len(c) < 5000:
        print(f"{symbol}: слишком мало свечей, пропуск")
        return
    pays = payouts(symbol, [15, 30, 60]) if payout else {}
    pool_events(symbol, c, ts)
    print(f"\n{symbol}: {len(c)} свечей. Доля угаданных на будущих кусках истории (4 куска).")
    for h, mnt in ((3, 15), (6, 30), (12, 60)):
        X, Y, Z = features(c, h, ts)
        res = walk(X, Y, Z, h)
        pay = pays.get(mnt)
        be = 1 / (1 + pay) if pay else None
        head = f"\nСрок {mnt} мин" + (f": выплата Deriv {100 * pay:.0f}%, порог безубыточности {100 * be:.1f}%" if be else
                                       " (выплата не запрошена; порог при 85% — 54.1%, при 80% — 55.6%)")
        print(head)
        for name, parts in res.items():
            n = sum(m for _, m in parts)
            w = sum(int(v) for v, _ in parts)
            if name.startswith("откат"):
                t = total.setdefault((name, mnt), [0, 0, []])
                t[0] += w
                t[1] += n
                if pay:
                    t[2].append(pay)
            if not n:
                print(f"  {name:<46} нет случаев")
                continue
            p = w / n
            ci = 1.96 * np.sqrt(p * (1 - p) / n)
            per = " ".join(f"{100 * v / m:.1f}" for v, m in parts if m)
            mark = "  ← выше порога с запасом" if be and p - ci > be else ""
            print(f"  {name:<46}{n:>7} ставок  {100 * p:5.1f}% ± {100 * ci:.1f}   [{per}]{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="frxGBPCAD", help="одна пара или несколько через запятую: frxEURUSD,frxUSDJPY,...")
    ap.add_argument("--candles", type=int, default=60000, help="сколько 5-минутных свечей скачать (60000 ≈ 10 месяцев торговли)")
    ap.add_argument("--payout", action="store_true", help="спросить у Deriv реальную выплату (нужен .env демо-счёта)")
    ap.add_argument("--delay", type=int, default=0, help="вход на столько 5-минутных свечей позже сигнала (проверка на «выбросы»)")
    ap.add_argument("--preset", choices=["fx"], help="готовый список пар: fx — 9 валютных пар, на которых проверяли правило отката")
    ap.add_argument("--list", action="store_true", help="показать символы Deriv (индексы, сырьё, валюты) и выйти")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.preset == "fx":
        a.symbol = "frxGBPNZD,frxEURNZD,frxGBPAUD,frxEURAUD,frxGBPJPY,frxEURJPY,frxEURUSD,frxGBPUSD,frxUSDJPY"
    if a.list:
        list_symbols()
        return
    global DELAY
    DELAY = a.delay
    if a.delay:
        print(f"ВХОД С ЗАДЕРЖКОЙ: через {5 * a.delay} мин после сигнала (по цене закрытия той свечи).")
    total = {}
    syms = a.symbol.split(",")
    for i, sym in enumerate(syms):
        if i and not a.synthetic:
            print("\n(пауза 60 с между парами, чтобы не упереться в лимит Deriv)")
            time.sleep(60)
        try:
            run_one(sym, a.candles, a.payout, a.synthetic, total)
        except SystemExit as e:
            print(f"{sym}: {e}")
    confidence_report()
    if len(syms) > 1:
        print("\nОБЩИЙ ИТОГ правил отката по всем парам:")
        for (name, mnt), (w, n, pays) in sorted(total.items()):
            if not n:
                continue
            p = w / n
            ci = 1.96 * np.sqrt(p * (1 - p) / n)
            be = 1 / (1 + float(np.mean(pays))) if pays else None
            tail = (f"; средний порог {100 * be:.1f}% → " + ("ВЫШЕ порога с запасом" if p - ci > be else
                    "не отличить от порога или ниже")) if be else ""
            print(f"  {name[7:]:<34} {mnt} мин: {n} ставок, угадано {100 * p:.1f}% ± {100 * ci:.1f}{tail}")
    print("\nКак читать: чтобы ставки были в плюсе, «угадано» должно быть ВЫШЕ порога безубыточности с запасом больше погрешности"
          "\n(±) и держаться во всех кусках. Окна ставок перекрываются, поэтому настоящая погрешность ещё больше напечатанной.")


if __name__ == "__main__":
    main()
