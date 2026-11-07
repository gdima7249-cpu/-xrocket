"""Тест стратегии «покупка просадки» по МНОГИМ парам сразу (спот, 1h):

    python dip_backtest.py --days 120 --max-pairs 20 [--synthetic]

Сделки со всех пар объединяются (сотни сделок — вывод надёжнее, чем по одной паре). Показано: средний итог
сделки в % после комиссий и проскальзывания, винрейт, и то же самое для первой и второй половин истории,
плюс контроль на перемешанных данных (если на случайных данных результат такой же — это не стратегия).
"""
import argparse
from dataclasses import replace

import ccxt
import numpy as np

import config
from config import Params
from data import fetch_history, synthetic
from exchanges import make_exchange
from optimize import shuffled
from strategies import dip

GRID = [(lb, d, tp, sl, cf) for lb in (24,) for d in (0.04, 0.06, 0.08)
        for tp, sl in ((0.03, 0.03), (0.05, 0.05)) for cf in (False, True)]


def trades_for(df, p, lookback, drop, take, stop, confirm, max_hold):
    """Список (индекс входа, итог сделки в долях) для одной пары."""
    en = dip(df, lookback, drop, confirm).values
    hi, lo, cl = df.high.values, df.low.values, df.close.values
    out, pos = [], None
    for i in range(len(df)):
        if pos:
            ei, entry = pos
            ex_price = None
            if lo[i] <= entry * (1 - stop):
                ex_price = entry * (1 - stop)
            elif hi[i] >= entry * (1 + take):
                ex_price = entry * (1 + take)
            elif i - ei >= max_hold:
                ex_price = cl[i]
            if ex_price is not None:
                ex_price *= 1 - p.slippage
                out.append((ei, ex_price * (1 - p.fee) / (entry * (1 + p.fee)) - 1))
                pos = None
        if pos is None and en[i]:
            pos = (i, cl[i] * (1 + p.slippage))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--max-hold", type=int, default=48, help="таймаут, часов")
    ap.add_argument("--null-runs", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    p = Params()
    print(f"Издержки: комиссия {p.fee*100:.2f}%/сторона, проскальзывание {p.slippage*100:.2f}%")

    if a.synthetic:
        data = {f"P{i}-USDT": synthetic(3000, seed=i + 1, drift=0.0) for i in range(a.max_pairs)}
    else:
        ex = make_exchange(config.EXCHANGE)
        syms = [s["symbol"] for s in ex.list_symbols()
                if s["symbol"].endswith("-USDT") and s.get("enableTrading", True)][: a.max_pairs]
        data = {}
        for s in syms:
            try:
                df = fetch_history(ex, s, "1h", a.days)
            except ccxt.ExchangeError as e:
                print(f"{s}: пропущено ({str(e)[:60]})")
                continue
            if len(df) >= 400:
                data[s] = df
                print(f"{s}: {len(df)} свечей")
    if not data:
        raise SystemExit("Нет данных.")
    print(f"\nПар: {len(data)}")

    rows = []
    for lb, drop, take, stop, cf in GRID:
        halves = [[], []]      # первая и вторая половины истории каждой пары
        allr = []
        for df in data.values():
            tr = trades_for(df, p, lb, drop, take, stop, cf, a.max_hold)
            allr += [t[1] for t in tr]
            halves[0] += [t[1] for t in tr if t[0] < len(df) * 0.5]
            halves[1] += [t[1] for t in tr if t[0] >= len(df) * 0.5]
        nullr = []
        for k in range(a.null_runs):
            for j, df in enumerate(data.values()):
                nullr += [t[1] for t in trades_for(shuffled(df, 100 * k + j), p, lb, drop, take, stop, cf, a.max_hold)]
        f = lambda v: (len(v), np.mean(v) * 100 if v else 0.0, np.mean(np.array(v) > 0) * 100 if v else 0.0)
        rows.append(((lb, drop, take, stop, cf), f(allr), f(halves[0]), f(halves[1]), f(nullr)))

    rows.sort(key=lambda r: r[1][1], reverse=True)
    print("\nПросадка от максимума за 24ч -> покупка; выход: тейк/стоп/таймаут. Итог — средний % на сделку после издержек.")
    print(f"{'падение':<8}{'тейк/стоп':<11}{'зелёная':<9}{'сделок':>7}{'ср.%':>7}{'win%':>6} |{'1-я пол.%':>10}{'2-я пол.%':>10} |{'контроль ср.%':>14}")
    for (lb, d, tp, sl, cf), tot, h1, h2, nl in rows:
        print(f"{d*100:>5.0f}%  {tp*100:>3.0f}/{sl*100:<3.0f}%   {'да' if cf else 'нет':<9}{tot[0]:>7}{tot[1]:>7.2f}{tot[2]:>6.0f} |"
              f"{h1[1]:>10.2f}{h2[1]:>10.2f} |{nl[1]:>14.2f}")
    good = [r for r in rows if r[1][0] >= 100 and r[2][1] > 0 and r[3][1] > 0]
    print(f"\nНаборов с ≥100 сделок и плюсом в обеих половинах истории: {len(good)} из {len(rows)}.")
    print("Сравните «ср.%» с колонкой «контроль»: если на перемешанных данных результат такой же, это не стратегия.")
