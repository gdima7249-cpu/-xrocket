"""Живой журнал «вперёд» для трендовых шортов (donchian_ls, ema_ls) — без сервера и без процессов, которые нужно держать.

Правила заморожены в файле forward_ls.json в день старта (дата, монеты, стратегии, сторона, издержки). Дальше в любой день
запускаете эту программу: она скачивает свежие свечи Binance, заново прогоняет ТЕ ЖЕ правила и показывает только сделки,
которые открылись ПОСЛЕ даты старта. Это честная проверка на новых данных: правила не менялись, прошлое не подгонялось.

  python forward_ls.py            # первый запуск создаёт старт на сегодня; дальше — отчёт
  python forward_ls.py --reset    # начать заново с сегодняшнего дня (старый файл будет перезаписан)
  python forward_ls.py --start 2026-10-01   # задать дату старта (для проверки)

Читайте отчёт раз в несколько дней. Выводы делать после 100+ закрытых сделок и нескольких недель, не раньше.
Открытые (ещё не закрытые) позиции в отчёте не показываются: сделка считается, когда закрылась по тейку, стопу, сигналу или таймауту.
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

from replay import PAIRS, download
from replay_ls import STRATS, make_synth, net, portfolio, run_all

FILE = "forward_ls.json"


def load_or_create(a):
    if os.path.exists(FILE) and not a.reset and not a.start:
        return json.load(open(FILE))
    start = (datetime.strptime(a.start, "%Y-%m-%d").timestamp() if a.start else time.time())
    cfg = {"start": start, "pairs": a.pairs.split(","), "side": a.side, "strats": a.strats.split(","),
           "fee": a.fee, "slip": a.slip}
    json.dump(cfg, open(FILE, "w"), ensure_ascii=False, indent=1)
    print(f"Старт зафиксирован: {datetime.fromtimestamp(start):%Y-%m-%d %H:%M}, монет {len(cfg['pairs'])}, "
          f"стороны: {cfg['side']}, стратегии: {', '.join(cfg['strats'])}, издержки круга {200 * (cfg['fee'] + cfg['slip']):.2f}%.")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--start", default="")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--side", default="short", choices=["short", "both", "long"])
    ap.add_argument("--strats", default="donchian_ls,ema_ls")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--pos-pct", type=float, default=0.05)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--synthetic", action="store_true", help="проверка кода без интернета")
    a = ap.parse_args()
    cfg = load_or_create(a)

    days_since = int((time.time() - cfg["start"]) / 86400) + 1
    need = days_since + 150                              # запас на прогрев индикаторов
    frames = {}
    for i, s in enumerate(cfg["pairs"]):
        try:
            frames[s] = make_synth(need, i, False) if a.synthetic else download(s + "USDT", "4h", need)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: пропуск ({type(e).__name__}: {e})")
    if not frames:
        raise SystemExit("Нет данных")
    if a.synthetic:
        t_ref = next(iter(frames.values()))["time"].to_numpy()
        for s_ in frames:
            frames[s_]["time"] = t_ref[:len(frames[s_])]

    closed = [c for c in run_all(frames, cfg["fee"], cfg["slip"], cfg["side"], cfg["strats"]) if c["t0"] >= cfg["start"]]
    last_candle = max(f["time"].iloc[-1] for f in frames.values()) / 1000
    print(f"\nЖурнал вперёд: прошло {(time.time() - cfg['start']) / 86400:.1f} дн. со старта; последняя свеча в данных: "
          f"{datetime.fromtimestamp(last_candle):%Y-%m-%d %H:%M}; закрытых сделок {len(closed)}.")
    if not closed:
        print("Закрытых сделок пока нет — это нормально в первые дни. Запустите позже.")
        return

    print(f"\n  {'стратегия':<14}{'сделок':>7}{'win%':>6}{'ср.%':>8}{'итого%':>9}")
    for st in cfg["strats"]:
        r = np.array([c["ret"] * 100 for c in closed if c["s"] == st])
        if len(r):
            print(f"  {st:<14}{len(r):>7}{100 * (r > 0).mean():>6.0f}{r.mean():>+8.2f}{r.sum():>+9.1f}")
    r_all = np.array([c["ret"] * 100 for c in closed])
    print(f"  {'все вместе':<14}{len(r_all):>7}{100 * (r_all > 0).mean():>6.0f}{r_all.mean():>+8.2f}{r_all.sum():>+9.1f}")

    by_week = {}
    for c in closed:
        by_week.setdefault(datetime.fromtimestamp(c["t0"]).strftime("%G-W%V"), []).append(c["ret"] * 100)
    print("\n  по неделям входа (сделок / ср.% на сделку):")
    for w in sorted(by_week):
        v = by_week[w]
        print(f"    {w}: {len(v):>3} / {np.mean(v):+.2f}")

    eq, curve, taken, skipped = portfolio(closed, a.pos_pct, a.max_pos)
    peak, dd = 1000.0, 0.0
    for _, v in curve:
        peak = max(peak, v)
        dd = min(dd, v / peak - 1)
    print(f"\n  счёт (старт 1000, {100 * a.pos_pct:.0f}% на сделку, до {a.max_pos} позиций): {eq:.0f}  ({100 * (eq / 1000 - 1):+.1f}%), "
          f"макс. просадка по закрытым {100 * dd:.1f}%, взято {taken}/пропущено {skipped}")
    print("\n  ориентир из истории (шорты, 7 лет): около +0.6…+1.0% на сделку, win 27–30%, просадки 20–30%.")
    if len(closed) < 100:
        print(f"  Пока слишком мало сделок ({len(closed)} из 100+) — выводов делать нельзя.")


if __name__ == "__main__":
    main()
