"""Оффлайн-проверка: можно ли угадывать «вверх/вниз» лучше порога безубыточности.

Берёт минутные свечи Binance (публично, ключи не нужны), строит признаки
(доходности, RSI, волатильность, объём, поток покупок taker, положение в диапазоне,
время суток), обучает логистическую регрессию и проверяет её только на
БУДУЩИХ данных (walk-forward: обучение на прошлом, проверка на следующем куске).
Рядом — простые правила (momentum, reversal, taker) и контроль на перемешанных ответах.

Порог безубыточности при выплате 85% — 54.05%. Вывод PASS только если винрейт
выше порога статистически значимо И в большинстве кусков времени.

Запуск:
  python binary_research.py                       # 12 пар, 30 дней, горизонты 1 и 5 мин
  python binary_research.py --days 60 --horizons 1,3,5
  python binary_research.py --synthetic           # проверка кода без интернета (случайные данные)
  python binary_research.py --gbm                 # + градиентный бустинг (pip install scikit-learn)
  python binary_research.py --synthetic --plant   # то же, но с вшитым «преимуществом» — детектор должен его найти
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

PAYOUT = 0.85
BREAKEVEN = 1 / (1 + PAYOUT)            # 0.5405
API = "https://data-api.binance.vision/api/v3/klines"
CACHE = "research_cache"
PAIRS = "BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,LTC,TRX,DOT".split(",")


# ---------------------------------------------------------------- данные
def download(symbol: str, days: int) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{symbol}_{days}d.csv"
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 6 * 3600:
        return pd.read_csv(path, parse_dates=["time"])
    end = int(time.time() * 1000)
    since = end - days * 86400 * 1000
    rows = []
    while since < end:
        r = requests.get(API, params={"symbol": symbol, "interval": "1m",
                                      "startTime": since, "limit": 1000}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 60_000
        time.sleep(0.05)
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
    df.columns = ["time", "open", "high", "low", "close", "volume", "taker_buy"]
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("time").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def synthetic(n: int, seed: int, plant: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    taker = np.clip(0.5 + rng.normal(0, 0.12, n), 0.02, 0.98)
    noise = rng.normal(0, 0.0007, n)
    ret = noise.copy()
    if plant:                                   # вшитое преимущество: поток покупок прошлой минуты
        ret[1:] += 0.0006 * (taker[:-1] - 0.5) / 0.12
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.0004, n))
    vol = rng.uniform(5, 50, n)
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "open": open_, "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread), "close": close,
        "volume": vol, "taker_buy": vol * taker})


# ---------------------------------------------------------------- признаки
FEATURES = ["r1", "r3", "r5", "r15", "r60", "rsi", "volat", "vol_ratio", "body",
            "taker", "taker5", "range_pos", "hsin", "hcos"]


def features(df: pd.DataFrame, horizon: int, delay: int = 0) -> pd.DataFrame:
    c = df["close"]
    lr = np.log(c).diff()
    vol60 = lr.rolling(60).std()
    f = pd.DataFrame({"time": df["time"], "close": c})
    for n in (1, 3, 5, 15, 60):
        f[f"r{n}"] = np.log(c).diff(n) / (vol60 * np.sqrt(n) + 1e-12)   # доходность в единицах волатильности
    d = c.diff()
    up, down = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    f["rsi"] = (100 - 100 / (1 + up / (down + 1e-12)) - 50) / 50
    f["volat"] = lr.rolling(15).std() / (vol60 + 1e-12) - 1
    f["vol_ratio"] = np.log1p(df["volume"] / (df["volume"].rolling(60).mean() + 1e-12))
    f["body"] = (c - df["open"]) / (df["high"] - df["low"] + 1e-12)
    f["taker"] = df["taker_buy"] / (df["volume"] + 1e-12) - 0.5
    f["taker5"] = f["taker"].rolling(5).mean()
    lo, hi = df["low"].rolling(20).min(), df["high"].rolling(20).max()
    f["range_pos"] = (c - lo) / (hi - lo + 1e-12) - 0.5
    minute = df["time"].dt.hour * 60 + df["time"].dt.minute
    f["hsin"], f["hcos"] = np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440)
    # delay: ставка открывается не по цене сигнальной свечи, а на delay минут позже (задержка исполнения/котировки)
    base, fut = c.shift(-delay), c.shift(-(delay + horizon))
    f["y"] = np.where(fut > base, 1.0, np.where(fut < base, 0.0, np.nan))   # ничья -> NaN (ставка возвращается)
    f["row"] = np.arange(len(f))
    return f.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["y"])


# ---------------------------------------------------------------- модель (numpy)
def fit_logreg(X, y, l2=1.0, iters=12):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    reg = l2 * np.eye(Z.shape[1]); reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg @ w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + reg + 1e-6 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return mu, sd, w


def predict(model, X):
    mu, sd, w = model
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), (X - mu) / sd] @ w, -30, 30)))


# ---------------------------------------------------------------- проверка
def z_vs_breakeven(wins, n):
    return (wins / n - BREAKEVEN) / np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n) if n else float("nan")


def make_gbm():
    """Градиентный бустинг (нелинейная модель; нужен scikit-learn). Умеренная глубина и сильная регуляризация против подгонки."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=4, max_iter=120, learning_rate=0.05, l2_regularization=5.0,
                                          min_samples_leaf=500, early_stopping=False, random_state=0)


def walk_forward(data: pd.DataFrame, horizon: int, folds: int, seed: int = 0, gbm: bool = False):
    """Возвращает {имя: [(выигрыши, ставки) по каждому тестовому куску]}."""
    t = data["time"]
    edges = pd.to_datetime(np.linspace(t.min().value, t.max().value, folds + 1))
    chunk = np.clip(np.searchsorted(edges, t.values, side="right") - 1, 0, folds - 1)
    X, y = data[FEATURES].to_numpy(), data["y"].to_numpy()
    rng = np.random.default_rng(seed)
    names = ["logreg", "logreg_уверенные20%"] + (["gbm", "gbm_уверенные20%"] if gbm else []) + [
        "momentum", "reversal", "taker_flow", "КОНТРОЛЬ(перемешано)"] + (["КОНТРОЛЬ_gbm(перемешано)"] if gbm else [])
    out = {k: [] for k in names}
    for k in range(1, folds):                       # обучение: куски < k, проверка: кусок k
        te = chunk == k
        tr = chunk < k
        # «зазор» горизонта между обучением и проверкой, чтобы ответы обучения не заглядывали в тест
        gap_start = t[te].min() - pd.Timedelta(minutes=horizon + 1)
        tr &= (t < gap_start).to_numpy()
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        model = fit_logreg(X[tr], y[tr])
        p_te = predict(model, X[te])
        p_tr = predict(model, X[tr])
        side = (p_te > 0.5).astype(float)
        out["logreg"].append(((side == y[te]).sum(), te.sum()))
        thr = np.quantile(np.abs(p_tr - 0.5), 0.80)               # порог уверенности выбран по ОБУЧЕНИЮ
        m = np.abs(p_te - 0.5) >= thr
        out["logreg_уверенные20%"].append(((side[m] == y[te][m]).sum(), int(m.sum())))
        f = data[te]
        for name, sig in (("momentum", f["r1"].to_numpy()), ("reversal", -f["r1"].to_numpy()),
                          ("taker_flow", f["taker5"].to_numpy())):
            mm = sig != 0
            out[name].append((((sig[mm] > 0).astype(float) == y[te][mm]).sum(), int(mm.sum())))
        if gbm:
            for tag, labels in (("gbm", y[tr]), ("КОНТРОЛЬ_gbm(перемешано)", rng.permutation(y[tr]))):
                m_ = make_gbm().fit(X[tr], labels)
                p_te_g = m_.predict_proba(X[te])[:, 1]
                side_g = (p_te_g > 0.5).astype(float)
                out[tag].append(((side_g == y[te]).sum(), te.sum()))
                if tag == "gbm":
                    thr_g = np.quantile(np.abs(m_.predict_proba(X[tr])[:, 1] - 0.5), 0.80)   # порог уверенности — по ОБУЧЕНИЮ
                    mg = np.abs(p_te_g - 0.5) >= thr_g
                    out["gbm_уверенные20%"].append(((side_g[mg] == y[te][mg]).sum(), int(mg.sum())))
        ctrl = fit_logreg(X[tr], rng.permutation(y[tr]))
        side_c = (predict(ctrl, X[te]) > 0.5).astype(float)
        out["КОНТРОЛЬ(перемешано)"].append(((side_c == y[te]).sum(), te.sum()))
    return out


def report(name, parts):
    if not parts:
        return f"  {name:<24} нет данных", False
    wins = sum(int(w) for w, _ in parts)
    n = sum(int(m) for _, m in parts)
    rates = [w / m for w, m in parts if m]
    z = z_vs_breakeven(wins, n)
    good = sum(r > BREAKEVEN for r in rates)
    roi = (wins / n) * PAYOUT - (1 - wins / n)
    passed = z > 3.0 and good >= max(1, int(np.ceil(0.8 * len(rates))))
    line = (f"  {name:<24} ставок {n:>8}  винрейт {100 * wins / n:5.2f}%  ROI/ставку {100 * roi:+6.2f}%  "
            f"z={z:+5.1f}  кусков выше порога {good}/{len(rates)}  "
            f"[{' '.join(f'{100 * r:.1f}' for r in rates)}]  {'PASS' if passed else '-'}")
    return line, passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--horizons", default="1,5", help="сроки ставки в минутах через запятую")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--folds", type=int, default=6, help="число кусков времени (проверка на 2-м ... последнем)")
    ap.add_argument("--payout", type=float, default=0.85, help="выплата при выигрыше (0.92 = 92%%); порог безубыточности = 1/(1+выплата)")
    ap.add_argument("--delay", type=int, default=0, help="задержка входа в минутах после сигнала (проверка на реалистичность)")
    ap.add_argument("--gbm", action="store_true", help="добавить градиентный бустинг (нужен: pip install scikit-learn; дольше)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plant", action="store_true", help="с --synthetic: вшить преимущество (проверка детектора)")
    a = ap.parse_args()
    global PAYOUT, BREAKEVEN
    PAYOUT, BREAKEVEN = a.payout, 1 / (1 + a.payout)

    frames = {}
    for i, s in enumerate(a.pairs.split(",")):
        try:
            frames[s] = synthetic(a.days * 1440 if a.synthetic else 0, i, a.plant) if a.synthetic \
                else download(s + "USDT", a.days)
            print(f"{s}: {len(frames[s])} минутных свечей", flush=True)
        except Exception as e:                          # одна пара не должна ронять весь тест
            print(f"{s}: пропуск ({type(e).__name__}: {e})")
    if not frames:
        raise SystemExit("Нет данных (проверьте интернет/доступ к data-api.binance.vision)")

    any_pass = False
    for h in (int(x) for x in a.horizons.split(",")):
        data = pd.concat([features(df, h, a.delay).assign(pair=s) for s, df in frames.items()]).sort_values("time")
        data = data.reset_index(drop=True)
        print(f"\n=== срок ставки {h} мин; задержка входа {a.delay} мин; примеров {len(data)}; порог безубыточности {100 * BREAKEVEN:.2f}% ===")
        res = walk_forward(data, h, a.folds, gbm=a.gbm)
        for name, parts in res.items():
            line, ok = report(name, parts)
            print(line)
            any_pass |= ok and not name.startswith("КОНТРОЛЬ")
    print("\nВЕРДИКТ:", "есть кандидат — повторить на другом периоде/парах, потом живой виртуальный тест"
          if any_pass else "преимущества над порогом не найдено (контроль ~50% — тест честный)")
    print("PASS = z>3 против порога И не менее 80% кусков времени выше порога. Перебирается много"
          " вариантов, поэтому один PASS нужно подтверждать на новых данных.")


if __name__ == "__main__":
    main()
