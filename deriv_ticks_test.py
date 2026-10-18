"""Проверка предсказуемости цен Deriv по ПУБЛИЧНОЙ ленте тиков (токен/ключ не нужен, ставок нет, деньги не тратятся).

Скачивает историю тиков символа (Volatility 100, 1HZ100V, BTC/USD ... — любой из списка Deriv), строит признаки по последним тикам
и проверяет walk-forward (учимся на прошлом, проверяем на будущем): можно ли угадывать направление цены через N тиков лучше случайного
и лучше порога безубыточности (при выплате 95% — 51.3%).

  pip install scikit-learn websocket-client
  python deriv_ticks_test.py                         # R_100 (Volatility 100), 5 тиков вперёд
  python deriv_ticks_test.py --symbol 1HZ100V --ahead 5 --payout 0.95
  python deriv_ticks_test.py --symbol cryBTCUSD --ahead 5
  python deriv_ticks_test.py --synthetic             # проверка кода без интернета (случайные данные)

Ожидание для синтетических индексов (Volatility ...): ровно около 50% — цену генерирует случайный генератор.
"""
import argparse
import json
import time

import numpy as np

APP_ID = "1089"


def fetch_ticks(symbol, total, app_id=APP_ID):
    """Публичная история тиков (по 5000 за запрос, назад во времени). Возвращает массив цен в хронологическом порядке."""
    import websocket
    ws = websocket.create_connection(f"wss://ws.derivws.com/websockets/v3?app_id={app_id}", timeout=30)
    prices, end = [], "latest"
    try:
        while len(prices) < total:
            ws.send(json.dumps({"ticks_history": symbol, "adjust_start_time": 1, "count": min(5000, total - len(prices)),
                                "end": end, "style": "ticks"}))
            msg = json.loads(ws.recv())
            if msg.get("error"):
                raise SystemExit(f"Deriv API: {msg['error'].get('message')} (проверьте название символа)")
            h = msg["history"]
            p, t = h["prices"], h["times"]
            if not p:
                break
            prices = list(map(float, p)) + prices
            end = t[0] - 1
            time.sleep(0.2)
    finally:
        ws.close()
    return np.array(prices)


def synthetic(n, seed=1):
    rng = np.random.default_rng(seed)
    return 1000 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))


def make_xy(prices, ahead, lags=20):
    """Признаки: последние lags изменений цены (в долях волатильности) и скользящие суммы; ответ: цена через ahead тиков выше (1) или ниже (0)."""
    r = np.diff(np.log(prices))
    vol = np.std(r) + 1e-12
    n = len(r)
    X, y, idx = [], [], []
    for t in range(lags + 60, n - ahead):
        w = r[t - lags:t] / vol
        feats = list(w) + [w[-5:].sum(), w[-10:].sum(), w.sum(), r[t - 60:t].std() / vol]
        X.append(feats)
        fut = prices[t + 1 + ahead] - prices[t + 1]
        y.append(1.0 if fut > 0 else (0.0 if fut < 0 else np.nan))
        idx.append(t)
    X, y = np.array(X), np.array(y)
    ok = ~np.isnan(y)
    return X[ok], y[ok]


def walk_forward(X, y, folds=5, payout=0.95, ahead=5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(y)
    edges = np.linspace(0, n, folds + 1).astype(int)
    be = 1 / (1 + payout)
    rows = {"gbm": [], "gbm_уверенные20%": [], "всегда вверх": [], "противоположно последнему изменению": []}
    for k in range(1, folds):
        tr = slice(0, edges[k] - ahead - 1)                       # зазор между обучением и проверкой
        te = slice(edges[k], edges[k + 1])
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=5.0,
                                           min_samples_leaf=200, early_stopping=False, random_state=0).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        thr = np.quantile(np.abs(m.predict_proba(X[tr])[:, 1] - 0.5), 0.8)
        side = (p > 0.5).astype(float)
        conf = np.abs(p - 0.5) >= thr
        yt = y[te]
        rows["gbm"].append(((side == yt).sum(), len(yt)))
        rows["gbm_уверенные20%"].append(((side[conf] == yt[conf]).sum(), int(conf.sum())))
        rows["всегда вверх"].append((int((yt == 1).sum()), len(yt)))
        last = X[te][:, 19]                                       # последнее изменение (в единицах волатильности)
        rows["противоположно последнему изменению"].append((int(((last < 0).astype(float) == yt)[last != 0].sum()), int((last != 0).sum())))
    return rows, be


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="R_100")
    ap.add_argument("--ahead", type=int, default=5, help="через сколько тиков смотрим направление")
    ap.add_argument("--ticks", type=int, default=100000, help="сколько тиков скачать")
    ap.add_argument("--payout", type=float, default=0.95)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    prices = synthetic(a.ticks) if a.synthetic else fetch_ticks(a.symbol, a.ticks)
    print(f"Тиков {len(prices)} ({a.symbol}), смотрим направление через {a.ahead} тиков.")
    X, y = make_xy(prices, a.ahead)
    rows, be = walk_forward(X, y, payout=a.payout, ahead=a.ahead)
    print(f"Порог безубыточности при выплате {100 * a.payout:.0f}%: {100 * be:.2f}%. Проверка только на будущем (5 кусков):\n")
    print(f"  {'вариант':<38}{'ставок':>9}{'угадано':>9}  по кускам")
    for name, parts in rows.items():
        n = sum(m for _, m in parts)
        w = sum(int(v) for v, _ in parts)
        per = " ".join(f"{100 * v / m:.1f}" for v, m in parts if m)
        z = (w / n - be) / np.sqrt(be * (1 - be) / n) if n else float("nan")
        print(f"  {name:<38}{n:>9}{100 * w / n:>8.2f}%  [{per}]   z против порога {z:+.1f}")
    print("\nКак читать: на ЗАВЕДОМО случайных данных (проверено на 12 рядах) результат «уверенных» колеблется примерно от 49.3% до 51.8%,"
          " потому что окна ставок перекрываются; поэтому отклонения меньше 1.5–2 п.п. ничего не значат."
          "\nЕсли везде около 50%, то цена непредсказуема: на ней нельзя выигрывать ни умным, ни простым ботом.")


if __name__ == "__main__":
    main()
