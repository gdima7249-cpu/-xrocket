"""Что дала бы простая регулярная покупка крипты (DCA: одна и та же сумма каждую неделю) — честно, с просадками. Без ботов и прогнозов.

Берёт реальные цены Binance (публично, ключи не нужны) и считает для BTC, ETH и корзины из многих монет:
  сколько вложено, сколько стоит вложенное сейчас, во сколько раз выросло, годовая доходность (IRR),
  максимальная просадка стоимости портфеля, самое плохое окно «вложил — через год»,
  и сравнение «покупать каждую неделю» против «купить всё сразу в начале».

  python dca_research.py                       # 7 лет, неделя = 1 покупка
  python dca_research.py --days 1825           # последние 5 лет
  python dca_research.py --synthetic           # проверка кода без интернета

Не совет вкладывать. История не повторяется, у многих монет цена ушла в ноль или они исчезли с рынка (в списке только те, что есть сегодня —
это завышает результат корзины). Результат в рублях зависит ещё от курса.
"""
import argparse
import math

import numpy as np
import pandas as pd

from replay import download, synthetic as _synth_1h, resample4h

BASKET = "BTC,ETH,SOL,XRP,BNB,DOGE,ADA,LINK,AVAX,DOT,LTC,TRX,NEAR,ATOM,UNI,XLM,FIL,AAVE,ETC,BCH".split(",")
WEEK_BARS = 6 * 7                                     # 4-часовых свечей в неделе


def prices(symbols, days, synthetic):
    out = {}
    for i, s in enumerate(symbols):
        try:
            df = resample4h(_synth_1h(days, i)) if synthetic else download(s + "USDT", "4h", days)
            out[s] = pd.Series(df["close"].to_numpy(), index=pd.to_datetime(df["time"].to_numpy(), unit="ms"))
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {str(e)[:60]})")
    if synthetic and out:                                  # у искусственных рядов время отличается на миллисекунды — выравниваем
        ref = next(iter(out.values())).index
        out = {k: pd.Series(v.to_numpy()[:len(ref)], index=ref[:len(v)]) for k, v in out.items()}
    return pd.DataFrame(out).sort_index()


def irr(flows, times_years):
    """Годовая доходность по потокам денег (расходы отрицательные, итоговая стоимость положительная)."""
    lo, hi = -0.99, 10.0
    f = lambda r: sum(c / (1 + r) ** t for c, t in zip(flows, times_years))  # noqa: E731
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def dca(price: pd.Series, amount=1.0):
    """Покупаем на amount раз в неделю. Возвращает (вложено, стоимость сейчас, IRR, макс. просадка, худший год)."""
    p = price.dropna()
    buy_idx = np.arange(0, len(p), WEEK_BARS)
    units = np.zeros(len(p))
    cum_units, invested = 0.0, np.zeros(len(p))
    inv = 0.0
    bi = set(buy_idx.tolist())
    flows, times = [], []
    t0 = p.index[0]
    for i in range(len(p)):
        if i in bi:
            cum_units += amount / p.iloc[i]
            inv += amount
            flows.append(-amount)
            times.append((p.index[i] - t0).total_seconds() / (365.25 * 86400))
        units[i], invested[i] = cum_units, inv
    value = units * p.to_numpy()
    flows.append(value[-1])
    times.append((p.index[-1] - t0).total_seconds() / (365.25 * 86400))
    peak = np.maximum.accumulate(value)
    dd = np.min(np.where(peak > 0, value / np.where(peak > 0, peak, 1) - 1, 0))
    # худший год: изменение стоимости вложенного к этому моменту «за год» — через отношение стоимость/вложено
    ratio = np.where(invested > 0, value / np.where(invested > 0, invested, 1), np.nan)
    yr = 6 * 365
    worst = np.nanmin(ratio[yr:] / ratio[:-yr] - 1) if len(ratio) > yr else float("nan")
    return inv, value[-1], irr(flows, times), dd, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=2555)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    P = prices(BASKET, a.days, a.synthetic)
    if P.empty:
        raise SystemExit("Нет данных")
    print(f"Период: {P.index[0]:%Y-%m-%d} … {P.index[-1]:%Y-%m-%d}, монет {P.shape[1]}. Покупка раз в неделю на одну и ту же сумму.\n")
    print(f"{'вариант':<28}{'вложено':>9}{'сейчас':>9}{'x':>7}{'в год':>8}{'макс.просадка':>15}{'худшее «за год»':>17}")

    def row(name, series):
        inv, val, r, dd, worst = dca(series)
        print(f"{name:<28}{inv:>9.0f}{val:>9.0f}{val / inv:>7.2f}{100 * r:>7.0f}%{100 * dd:>14.0f}%{100 * worst:>16.0f}%")

    for s in ("BTC", "ETH"):
        if s in P:
            row(f"{s} (каждую неделю)", P[s])
    # корзина с учётом появления монет: равновесная доходность по доступным монетам
    ret = P.pct_change().mean(axis=1).fillna(0)
    basket = (1 + ret).cumprod()
    row("корзина 20 монет (равные доли)", basket)
    if "BTC" in P:
        p = P["BTC"].dropna()
        lump = p.iloc[-1] / p.iloc[0]
        print(f"\nДля сравнения: купить BTC один раз в начале периода: x{lump:.2f}; купить BTC в самый высокий момент периода: "
              f"сейчас {100 * (p.iloc[-1] / p.max() - 1):+.0f}% от того пика; от самого низкого: x{p.iloc[-1] / p.min():.2f}.")
        dd_series = p / p.cummax() - 1
        print(f"Самая глубокая просадка BTC за период: {100 * dd_series.min():.0f}% (в этом месте вложенное стоило бы в {1 / (1 + dd_series.min()):.1f} раза меньше пика).")
    print("\nВАЖНО: в корзине только монеты, которые есть на Binance сегодня, поэтому она выглядит лучше, чем была бы честная корзина "
          "(мёртвые проекты в неё не попали). История не гарантирует будущего. Это не совет вкладывать.")


if __name__ == "__main__":
    main()
