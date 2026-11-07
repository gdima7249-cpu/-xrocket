"""Крайние значения фандинга как сигнал направления: когда на фьючерсах все в лонгах (фандинг очень высокий) — дальше вниз?
Когда все в шортах (фандинг отрицательный) — дальше вверх? Ничего не покупает, ключи не нужны.

Данные: ставки фандинга Binance (раз в 8 часов) и 4-часовые свечи за 3 года (публично; кэш в research_cache).

Правило (без подглядывания в будущее):
  событие  — ставка фандинга в момент выплаты выше X (или ниже −Y), либо в верхних/нижних 5% своей истории за прошлые 90 дней;
  вход     — по цене открытия ближайшей 4-часовой свечи ПОСЛЕ выплаты, против толпы (высокий фандинг — шорт, отрицательный — лонг);
  выход    — через 1, 3 или 7 дней по закрытию;
  учтено   — фандинг, который позиция платит или получает, пока открыта (шорт при высоком фандинге его получает),
             комиссия 0.04% и проскальзывание 0.02% за сторону; сделки по одной монете не перекрываются.
Проверки: погрешность по дням, первая и вторая половина периода, сравнение со случайным входом в ту же сторону.

  python funding_research.py                    # 15 монет, 3 года
  python funding_research.py --days 1825 --coins BTC,ETH,SOL
  python funding_research.py --synthetic        # случайные данные без связи фандинга с ценой: эффекта быть не должно
"""
import argparse
import time

import numpy as np
import pandas as pd

from replay import download
from replay_ls import fetch_funding

COINS = "BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,DOT,LTC,TRX,NEAR,ATOM,UNI"
H4 = 4 * 3600


def synthetic(days, seed):
    rng = np.random.default_rng(seed)
    n = days * 6
    t = (int(time.time()) // H4) * H4 - n * H4 + np.arange(n) * H4
    close = 100 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.01))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"time": t * 1000.0, "open": open_, "close": close})
    ft = t[::2]                                                     # раз в 8 часов
    regime = np.sin(np.arange(len(ft)) / 150.0)
    fr = 0.0001 + 0.0002 * regime + rng.normal(0, 0.0001, len(ft))
    return df, (ft.astype(float), fr)


def trades(df, fund, rule, hold_days):
    """Сделки по одной монете. rule(rates, i) -> сторона (+1 лонг, −1 шорт, 0 — нет события).
    Возвращает (время, доходность без издержек с фандингом, движение цены сверх дрейфа рынка, сторона)."""
    ft, fr = np.asarray(fund[0], dtype=float), np.asarray(fund[1], dtype=float)
    cum = np.r_[0, np.cumsum(fr)]
    bt = df["time"].to_numpy() / 1000.0
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    hold = hold_days * 6
    drift = float(np.mean(c[hold - 1:] / o[:len(o) - hold + 1] - 1))     # средний результат лонга на тот же срок без сигнала
    T, R, X, S = [], [], [], []
    busy_until = -np.inf
    for i in range(len(ft)):
        if ft[i] < busy_until:
            continue
        side = rule(fr, i)
        if not side:
            continue
        j = np.searchsorted(bt, ft[i], side="left")                # первая свеча, открывшаяся не раньше выплаты
        if j + hold >= len(bt):
            break
        t0, t1 = bt[j], bt[j + hold - 1] + H4
        price = side * (c[j + hold - 1] / o[j] - 1)
        paid = cum[np.searchsorted(ft, t1, side="right")] - cum[np.searchsorted(ft, t0, side="right")]
        T.append(t0)
        R.append(price - side * paid)                              # лонг платит положительный фандинг, шорт получает
        X.append(price - side * drift)                             # движение цены сверх обычного дрейфа рынка (без фандинга)
        S.append(side)
        busy_until = t1
    return np.array(T), np.array(R), np.array(X), np.array(S)


def rule_abs(hi, lo):
    return lambda fr, i: -1 if fr[i] >= hi else (1 if fr[i] <= -lo else 0)


def rule_pct(q, window=270):                                       # 270 выплат = 90 дней
    def f(fr, i):
        if i < window:
            return 0
        past = fr[i - window:i]
        if fr[i] >= np.quantile(past, 1 - q):
            return -1
        if fr[i] <= np.quantile(past, q):
            return 1
        return 0
    return f


def day_t(times, net):
    d = pd.Series(net).groupby((np.asarray(times) // 86400).astype(np.int64)).sum()
    if len(d) < 2 or d.std() == 0:
        return float("nan"), len(d)
    return float(d.mean() / (d.std() / np.sqrt(len(d)))), len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default=COINS)
    ap.add_argument("--days", type=int, default=1095)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    cost = 2 * (a.fee + a.slip)

    data = {}
    for k, s in enumerate(a.coins.split(",")):
        try:
            if a.synthetic:
                data[s] = synthetic(a.days, k)
            else:
                data[s] = (download(s + "USDT", "4h", a.days), fetch_funding(s + "USDT", a.days))
            print(f"{s}: {len(data[s][0])} свечей, {len(data[s][1][0])} выплат фандинга")
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {str(e)[:80]})")
    if not data:
        raise SystemExit("Нет данных (проверьте доступ к fapi.binance.com и data-api.binance.vision)")
    mid = float(np.median(np.concatenate([d[0]["time"].to_numpy() / 1000.0 for d in data.values()])))

    rules = [("фандинг ≥0.05% / ≤−0.01%", rule_abs(0.0005, 0.0001)),
             ("фандинг ≥0.10% / ≤−0.03%", rule_abs(0.0010, 0.0003)),
             ("верхние/нижние 5% за 90 дн", rule_pct(0.05)),
             ("верхние/нижние 1% за 90 дн", rule_pct(0.01))]
    print(f"\nИздержки на сделку: {100 * cost:.2f}% + фандинг за время удержания (уже внутри доходности).")
    print("Против толпы: высокий фандинг — шорт, отрицательный — лонг. Доходности в % на сделку ПОСЛЕ издержек.")
    print("«сверх рынка» — движение цены в сторону сделки МИНУС то, что за тот же срок дал бы вход в ту же сторону без сигнала,")
    print("без фандинга и издержек: это и есть перевес сигнала. Если сигналы почти все лонги, а рынок рос, «в среднем» будет в плюсе")
    print("и без всякого перевеса — смотрите на «сверх рынка».\n")
    print(f"  {'событие':<28}{'срок':>6}{'сделок':>8}{'лонг/шорт':>11}{'угадано':>9}{'в среднем':>11}{'сверх рынка':>13}"
          f"{'t (дни)':>9}{'1-я пол.':>10}{'2-я пол.':>10}")
    total = good = 0
    for name, rule in rules:
        for hold in (1, 3, 7):
            T, R, X, S = [], [], [], []
            for df, fund in data.values():
                t, r, x, sd = trades(df, fund, rule, hold)
                T.append(t)
                R.append(r)
                X.append(x)
                S.append(sd)
            t, r, x, sd = (np.concatenate(v) for v in (T, R, X, S))
            if len(r) < 20:
                continue
            total += 1
            net = r - cost
            tt, nd = day_t(t, x)                                   # значимость — по перевесу сверх рынка
            h1, h2 = x[t < mid], x[t >= mid]
            m1 = h1.mean() if len(h1) else np.nan
            m2 = h2.mean() if len(h2) else np.nan
            ok = net.mean() > 0 and x.mean() > cost and m1 > 0 and m2 > 0 and tt > 3
            good += bool(ok)
            ls = f"{100 * (sd > 0).mean():.0f}%/{100 * (sd < 0).mean():.0f}%"
            print(f"  {name:<28}{str(hold) + 'д':>6}{len(net):>8}{ls:>11}{100 * (r > 0).mean():>8.1f}%{100 * net.mean():>10.3f}%"
                  f"{100 * x.mean():>12.3f}%{tt:>9.1f}{100 * m1:>9.3f}%{100 * m2:>9.3f}%" + ("  ← устойчивый плюс" if ok else ""))
    print(f"\nПроверено вариантов: {total}. Устойчивых (перевес сверх рынка больше издержек, в обеих половинах, t > 3): {good}.")
    print("«1-я пол.»/«2-я пол.» и t тоже считаются по перевесу сверх рынка. Фандинг, полученный шортом при высокой ставке, — реальный доход,"
          "\nно это не предсказание направления; он уже включён в «в среднем».")


if __name__ == "__main__":
    main()
