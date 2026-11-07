"""Риск правил A и B из rebound_paper.py на истории: насколько плохо бывает в худших случаях. Ничего не покупает.

Берёт те же 5-минутные свечи, что rebound_research.py (из research_cache, если уже скачаны), и для каждого правила показывает:
  * худшие сделки и долю сделок с убытком больше 3% / 5% / 10%;
  * худший день (сумма всех сделок дня — в обвал сигналы приходят сразу по многим монетам);
  * максимальную просадку, если на каждую сделку ставить одну и ту же сумму (в процентах от одной ставки);
  * итог по годам.
Издержки — те же 0.12% на сделку.

  python rebound_risk.py                                       # последний год, 10 монет
  python rebound_risk.py --days 730 --ago 1095                 # 2021–2023 (Luna, FTX)
  python rebound_risk.py --days 1095 --coins DOT,TRX,ATOM,UNI,XLM,ETC,BCH,FIL,NEAR
  python rebound_risk.py --synthetic
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import rebound_research as RR

RULES = {"A (≥2% за 5м)": (1, 0.02), "B (≥3% за 15м)": (3, 0.03)}
HOLD = 12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default=RR.COINS)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--ago", type=int, default=0)
    ap.add_argument("--cost", type=float, default=0.0012)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    data = {}
    for i, s in enumerate(a.coins.split(",")):
        try:
            data[s] = RR.synthetic_5m(a.days, i) if a.synthetic else RR.download_5m(s + "USDT", a.days, a.ago)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {str(e)[:80]})")
    if not data:
        raise SystemExit("Нет данных")
    print(f"Монет: {len(data)}, период {a.days} дней" + (f", закончился {a.ago} дн. назад" if a.ago else "") +
          f". Издержки {100 * a.cost:.2f}% на сделку. Все цифры — в процентах от ОДНОЙ ставки.\n")

    for name, (k, thr) in RULES.items():
        rows = []
        for s, df in data.items():
            t, r = RR.events(df, k, thr, HOLD, -1)
            rows += [(tt, s, rr - a.cost) for tt, rr in zip(t, r)]
        if not rows:
            print(f"{name}: сделок нет\n")
            continue
        d = pd.DataFrame(rows, columns=["t", "coin", "net"]).sort_values("t")
        d["date"] = pd.to_datetime(d["t"], unit="ms", utc=True)
        net = d["net"].to_numpy()
        eq = np.cumsum(net)
        dd = float(np.min(eq - np.maximum.accumulate(np.r_[0, eq])[1:]))
        day = d.groupby(d["date"].dt.date)["net"].sum()
        # самая длинная серия убыточных сделок подряд
        streak = best = 0
        for x in net:
            streak = streak + 1 if x < 0 else 0
            best = max(best, streak)
        print(f"Правило {name}: сделок {len(d)}, в плюсе {100 * (net > 0).mean():.0f}%, в среднем {100 * net.mean():+.3f}%, "
              f"сумма {100 * net.sum():+.0f}% от ставки")
        print(f"  Убыток больше 3%: {100 * (net < -0.03).mean():.1f}% сделок; больше 5%: {100 * (net < -0.05).mean():.1f}%; "
              f"больше 10%: {100 * (net < -0.10).mean():.2f}%")
        print(f"  Худший день: {100 * day.min():+.1f}% ({day.idxmin()}, сделок в этот день: {int((d['date'].dt.date == day.idxmin()).sum())}); "
              f"лучший день: {100 * day.max():+.1f}% ({day.idxmax()})")
        print(f"  Максимальная просадка суммы: {100 * dd:.0f}% от одной ставки; самая длинная серия убытков подряд: {best}")
        print("  Худшие сделки:")
        for _, r in d.nsmallest(5, "net").iterrows():
            print(f"    {r['date']:%Y-%m-%d %H:%M} {r['coin']:<5} {100 * r['net']:+.1f}%")
        yearly = d.groupby(d["date"].dt.year)["net"].agg(["count", "mean", "sum"])
        print("  По годам: " + "; ".join(f"{y}: {int(c)} сд., в ср. {100 * m:+.2f}%, сумма {100 * s:+.0f}%"
                                         for y, (c, m, s) in yearly.iterrows()))
        print()
    print("Как читать: «сумма от ставки» +40% значит, что при ставке 1000 руб. на каждую сделку за период вышло бы +400 руб."
          "\nПросадка −60% значит, что в худший момент сумма была на 600 руб. ниже своего максимума (при той же ставке)."
          "\nВ день обвала сигналы приходят сразу по многим монетам: если ставить на все, нужен запас на несколько ставок одновременно.")


if __name__ == "__main__":
    main()
