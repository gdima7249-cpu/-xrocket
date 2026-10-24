"""Оффлайн-проверка «ИИ-бота для фьючерсов» (лонг и шорт) на истории Binance, с теми же проверками, что и раньше.

Идея: модель (логистическая регрессия и градиентный бустинг) учится на ПРОШЛОМ предсказывать знак доходности на H свечей вперёд
по признакам (доходности, волатильность, RSI, положение в диапазоне, объём, расстояние до EMA, поведение BTC и сила монеты
относительно BTC). Торгуем только самые уверенные сигналы: p > 0.5 -> лонг, p < 0.5 -> шорт; держим ровно H свечей.
Проверка только на БУДУЩЕМ (walk-forward), издержки круга учтены, рядом контроль на перемешанных ответах,
t-статистика с кластерами по дням и расчёт по счёту (доля капитала на сделку, просадка, доходность по годам).

  pip install scikit-learn
  python ai_futures_research.py                      # 20 пар, 2555 дней, 4-часовые свечи, горизонт 6 свечей (сутки)
  python ai_futures_research.py --horizon 3 --top 0.2
  python ai_futures_research.py --synthetic [--plant]   # проверка кода без интернета; --plant вшивает закономерность

Не учтено: фандинг фьючерсов, ликвидации (плечо 1x), задержка исполнения. История ≠ будущее.
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from binary_research import fit_logreg, predict
from replay import PAIRS, download
from replay_ls import make_synth, portfolio, net, BAR_SEC

FEATS = ["r1", "r3", "r6", "r12", "r24", "r42", "volat", "rsi", "range_pos", "vol_ratio", "ema50", "ema200",
         "btc_r6", "btc_r24", "btc_r84", "btc_volat", "rel24"]


def _rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0).rolling(n).mean(), (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / (dn + 1e-12)) - 50) / 50


def build(frames: dict, btc: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Таблица признаков и ответов по всем парам. Все признаки на момент закрытия свечи t; вход по открытию t+1."""
    bc = btc.set_index("time")["close"]
    blr = np.log(bc).diff()
    btcf = pd.DataFrame({"btc_r6": np.log(bc).diff(6), "btc_r24": np.log(bc).diff(24), "btc_r84": np.log(bc).diff(84),
                         "btc_volat": blr.rolling(24).std() / (blr.rolling(120).std() + 1e-12) - 1})
    parts = []
    for sym, df in frames.items():
        df = df.reset_index(drop=True)
        c, o = df["close"], df["open"]
        lr = np.log(c).diff()
        vol = lr.rolling(24).std()
        f = pd.DataFrame({"time": df["time"], "sym": sym})
        for n in (1, 3, 6, 12, 24, 42):
            f[f"r{n}"] = np.log(c).diff(n) / (vol * np.sqrt(n) + 1e-12)
        f["volat"] = vol / (lr.rolling(120).std() + 1e-12) - 1
        f["rsi"] = _rsi(c)
        lo, hi = df["low"].rolling(42).min(), df["high"].rolling(42).max()
        f["range_pos"] = (c - lo) / (hi - lo + 1e-12) - 0.5
        f["vol_ratio"] = np.log1p(df["volume"] / (df["volume"].rolling(42).mean() + 1e-12))
        f["ema50"] = (c / c.ewm(span=50, adjust=False).mean() - 1) / (vol + 1e-12)
        f["ema200"] = (c / c.ewm(span=200, adjust=False).mean() - 1) / (vol + 1e-12)
        f = f.merge(btcf, left_on="time", right_index=True, how="left")
        f["rel24"] = f["r24"] - f["btc_r24"] / (vol.values * np.sqrt(24) + 1e-12)
        # сделка: вход по открытию t+1, выход по закрытию t+horizon (всего horizon свечей удержания)
        f["pe"] = o.shift(-1).values
        f["px"] = c.shift(-horizon).values
        f["t_exit"] = df["time"].shift(-horizon).values
        f["y"] = np.where(f["px"] > f["pe"], 1.0, 0.0)
        f["row"] = np.arange(len(f))
        parts.append(f.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS + ["pe", "px", "t_exit"]))
    return pd.concat(parts).sort_values("time").reset_index(drop=True)


def make_gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, l2_regularization=10.0,
                                          min_samples_leaf=800, early_stopping=False, random_state=0)


def cluster_t(returns_pct, days):
    """t-статистика среднего по сделкам; сделки одного дня зависимы -> ошибка считается по суммам отклонений в дне."""
    r = np.asarray(returns_pct)
    if len(r) < 5:
        return 0.0
    m = r.mean()
    sums = {}
    for x, d in zip(r - m, days):
        sums[d] = sums.get(d, 0.0) + x
    se = np.sqrt(sum(v * v for v in sums.values())) / len(r)
    return m / se if se > 0 else 0.0


def walk_forward(data, horizon, folds, top, fee, slip, models, seed=0):
    """Возвращает {модель: список сделок-словарей (side, ret, t0, t1, fold)} только по тестовым кускам."""
    t = data["time"].to_numpy()
    edges = np.linspace(t.min(), t.max(), folds + 1)
    chunk = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, folds - 1)
    X, y = data[FEATS].to_numpy(), data["y"].to_numpy()
    rng = np.random.default_rng(seed)
    out = {m: [] for m in models}
    out.update({f"КОНТРОЛЬ_{m}": [] for m in models})
    for k in range(1, folds):
        te = chunk == k
        tr = (chunk < k) & (t < t[te].min() - horizon * BAR_SEC * 1000)          # зазор, чтобы ответы обучения не заглядывали в тест
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        sel_test = te & (data["row"].to_numpy() % horizon == 0)                  # непересекающиеся входы по каждой паре
        for name in models:
            for tag, labels in ((name, y[tr]), (f"КОНТРОЛЬ_{name}", rng.permutation(y[tr]))):
                if name == "logreg":
                    mdl = fit_logreg(X[tr], labels)
                    p_tr, p_te = predict(mdl, X[tr]), predict(mdl, X[sel_test])
                else:
                    mdl = make_gbm().fit(X[tr], labels)
                    p_tr, p_te = mdl.predict_proba(X[tr])[:, 1], mdl.predict_proba(X[sel_test])[:, 1]
                thr = np.quantile(np.abs(p_tr - 0.5), 1 - top)                    # порог уверенности выбран по ОБУЧЕНИЮ
                rows = data[sel_test].reset_index(drop=True)
                take = np.abs(p_te - 0.5) >= thr
                for i in np.where(take)[0]:
                    side = 1 if p_te[i] > 0.5 else -1
                    r = rows.iloc[i]
                    out[tag].append({"side": side, "ret": net(side, r["pe"], r["px"], fee, slip), "fold": k,
                                     "t0": r["time"] / 1000 + BAR_SEC, "t1": r["t_exit"] / 1000 - BAR_SEC, "sym": r["sym"]})
    return out


def summary(name, trades):
    if not trades:
        return f"  {name:<28} нет сделок", None
    r = np.array([x["ret"] * 100 for x in trades])
    days = [datetime.fromtimestamp(x["t0"]).date() for x in trades]
    t = cluster_t(r, days)
    folds = sorted({x["fold"] for x in trades})
    fm = [np.mean([x["ret"] * 100 for x in trades if x["fold"] == f]) for f in folds]
    longs = [x["ret"] * 100 for x in trades if x["side"] > 0]
    shorts = [x["ret"] * 100 for x in trades if x["side"] < 0]
    line = (f"  {name:<28}{len(r):>7} сделок  win {100 * (r > 0).mean():>3.0f}%  ср. {r.mean():+6.2f}%  t={t:+5.1f}  "
            f"лонги {np.mean(longs) if longs else 0:+5.2f} ({len(longs)})  шорты {np.mean(shorts) if shorts else 0:+5.2f} ({len(shorts)})  "
            f"по кускам [{' '.join(f'{v:+.2f}' for v in fm)}]")
    return line, (r.mean(), t, sum(v > 0 for v in fm), len(fm))


def account(trades, pos_pct, max_pos):
    eq, curve, taken, skipped = portfolio(trades, pos_pct, max_pos)
    if len(curve) < 2:
        return "    счёт: мало сделок"
    years = (curve[-1][0] - curve[0][0]) / (365.25 * 86400)
    peak, dd, by_year = 1000.0, 0.0, {}
    for tt, v in curve:
        peak = max(peak, v)
        dd = min(dd, v / peak - 1)
        by_year[datetime.fromtimestamp(tt).year] = v
    prev, parts = 1000.0, []
    for yr in sorted(by_year):
        parts.append(f"{yr}: {100 * (by_year[yr] / prev - 1):+.0f}%")
        prev = by_year[yr]
    cagr = (eq / 1000) ** (1 / years) - 1 if eq > 0 else -1
    return (f"    счёт: {eq / 1000:.2f}x, {100 * cagr:+.0f}% в год, макс. просадка {100 * dd:.0f}%, "
            f"взято {taken}/пропущено {skipped};  " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=2555)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--horizon", type=int, default=6, help="горизонт удержания в 4-часовых свечах (6 = сутки)")
    ap.add_argument("--top", type=float, default=0.3, help="доля самых уверенных сигналов, по которым торгуем")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--pos-pct", type=float, default=0.05)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--no-gbm", action="store_true", help="без градиентного бустинга (быстрее, не нужен scikit-learn)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--plant", action="store_true")
    a = ap.parse_args()

    syms = a.pairs.split(",")
    frames = {}
    for i, s in enumerate(syms):
        try:
            frames[s] = make_synth(a.days, i, a.plant) if a.synthetic else download(s + "USDT", "4h", a.days)
            print(f"{s}: {len(frames[s])} 4-часовых свечей", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {e})")
    if not frames:
        raise SystemExit("Нет данных")
    if a.synthetic:                                  # у искусственных рядов время отличается на миллисекунды — выравниваем
        t_ref = next(iter(frames.values()))["time"].to_numpy()
        for s_ in frames:
            frames[s_]["time"] = t_ref[:len(frames[s_])]
    if "BTC" in frames:
        btc = frames["BTC"]
    else:
        btc = make_synth(a.days, 0, a.plant) if a.synthetic else download("BTCUSDT", "4h", a.days)

    data = build(frames, btc, a.horizon)
    models = ["logreg"] + ([] if a.no_gbm else ["gbm"])
    print(f"\nПримеров {len(data)}, горизонт {a.horizon} св. ({a.horizon * 4} ч), торгуем топ-{100 * a.top:.0f}% уверенных сигналов, "
          f"издержки круга {200 * (a.fee + a.slip):.2f}%, расчёт по счёту: {100 * a.pos_pct:.0f}% на сделку, до {a.max_pos} позиций.\n")
    res = walk_forward(data, a.horizon, a.folds, a.top, a.fee, a.slip, models)

    verdicts = {}
    for name, trades in res.items():
        line, v = summary(name, trades)
        print(line)
        if not name.startswith("КОНТРОЛЬ") and trades:
            print(account(trades, a.pos_pct, a.max_pos))
        verdicts[name] = v

    print("\nВЕРДИКТ (всё сразу: ср.% > 0, t>3, плюс не менее чем в 4 из 5 кусков времени, лучше контроля минимум на 0.3 п.п.):")
    any_ok = False
    for m in models:
        v, c = verdicts.get(m), verdicts.get(f"КОНТРОЛЬ_{m}")
        if not v:
            print(f"  {m:<8} нет сделок"); continue
        ok = v[0] > 0 and v[1] > 3 and v[2] >= max(1, round(0.8 * v[3])) and v[0] - (c[0] if c else 0) >= 0.3
        any_ok |= ok
        print(f"  {m:<8} ср.% {v[0]:+.2f}, t={v[1]:+.1f}, плюсовых кусков {v[2]}/{v[3]}, "
              f"лучше контроля на {v[0] - (c[0] if c else 0):+.2f} -> {'КАНДИДАТ' if ok else 'не проходит'}")
    print("\nКАНДИДАТ ≠ деньги: проверьте на других монетах/периоде, учтите фандинг, затем живой виртуальный журнал."
          if any_ok else "\nПреимущества не найдено (контроль ~0 — тест честный).")


if __name__ == "__main__":
    main()
