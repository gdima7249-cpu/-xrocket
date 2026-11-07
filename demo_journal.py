"""Журнал ваших ставок на ДЕМО-счёте с честной статистикой: отличается ли ваш результат от случайного и от порога безубыточности.

Вы открываете сделки сами (на любой площадке, в том числе OTC), а сюда записываете итог одной буквой. Программа ничего не подключает
и ни к чему не обращается — это просто калькулятор по вашим записям.

  python demo_journal.py                    # интерактивно: w (выигрыш), l (проигрыш), d (ничья), stat, undo, q
  python demo_journal.py --payout 0.92      # выплата площадки (по умолчанию 0.92)
  python demo_journal.py --selftest

Команды (можно добавить пометку после буквы, например:  w audcad otc up):
  w / l / d    записать выигрыш / проигрыш / ничью (ничья: ставка возвращается, в статистику не входит)
  stat         сводка: сколько нужно ставок, чтобы увидеть разницу, и что показывает ваш результат
  undo         удалить последнюю запись     q  выход
Записи хранятся в demo_journal.csv.
"""
import argparse
import csv
import math
import os
from datetime import datetime

FILE = "demo_journal.csv"


def load():
    if not os.path.exists(FILE):
        return []
    with open(FILE, encoding="utf-8") as f:
        return [r for r in csv.reader(f) if r]


def save(rows):
    with open(FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def wilson(w, n, z=1.96):
    """95%-доверительный интервал для доли выигрышей."""
    if n == 0:
        return 0.0, 1.0
    p = w / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den, (c + r) / den


def longest(seq, val):
    best = cur = 0
    for x in seq:
        cur = cur + 1 if x == val else 0
        best = max(best, cur)
    return best


def stats(results, payout):
    """results: список 'w'/'l'/'d'. Возвращает текст со сводкой."""
    seq = [r for r in results if r in ("w", "l")]
    n, w = len(seq), sum(r == "w" for r in seq)
    be = 1 / (1 + payout)
    if n == 0:
        return "Пока нет записанных ставок."
    p = w / n
    lo, hi = wilson(w, n)
    roi = p * payout - (1 - p)
    z = (p - be) / math.sqrt(be * (1 - be) / n)
    lines = [f"Ставок (без ничьих) {n}: выигрышей {w} ({100 * p:.1f}%), проигрышей {n - w}; ничьих {results.count('d')}.",
             f"Порог безубыточности при выплате {100 * payout:.0f}%: {100 * be:.2f}%. Ваш ROI на ставку: {100 * roi:+.1f}%.",
             f"95%-интервал вашего винрейта: {100 * lo:.1f}% … {100 * hi:.1f}%.",
             f"Самая длинная серия выигрышей подряд: {longest(seq and [r for r in seq], 'w')}, проигрышей: {longest(seq, 'l')}."]
    if lo > be:
        lines.append("Вывод: даже нижняя граница выше порога — результат статистически выше безубыточности (но проверьте, что это не подгонка и не демо-эффект).")
    elif hi < be:
        lines.append("Вывод: верхняя граница ниже порога — на таком результате вы теряете деньги.")
    else:
        lines.append(f"Вывод: пока нельзя отличить ваш результат от порога ({100 * be:.2f}%) — выводов делать рано.")
    # сколько ставок нужно, чтобы разглядеть разницу
    for d in (0.02, 0.05):
        need = math.ceil((1.96 * math.sqrt(0.25) / d) ** 2)
        lines.append(f"  чтобы уверенно заметить отклонение винрейта на {100 * d:.0f} п.п., нужно около {need} ставок (у вас {n}).")
    # как часто случайные ставки дают такую же серию
    if n >= 10:
        lines.append(f"  для сравнения: при угадывании 50% серии по 6–8 побед подряд случаются регулярно; "
                     f"ожидаемая максимальная серия на {n} ставках около {math.log2(n):.0f}.")
    return "\n".join(lines)


def selftest():
    ok = []
    ok.append(("порог при 92% = 52.08%", abs(1 / 1.92 - 0.5208) < 1e-4))
    t = stats(list("w" * 6 + "l" * 4), 0.92)
    ok.append(("60% угадано -> ROI +15.2%", "60.0%" in t and "+15.2%" in t))
    ok.append(("малая выборка: нельзя отличить от порога", "нельзя отличить" in t))
    t2 = stats(list("w" * 700 + "l" * 300), 0.92)
    ok.append(("70% на 1000 ставках: нижняя граница выше порога", "статистически выше" in t2))
    t3 = stats(list("w" * 450 + "l" * 550), 0.92)
    ok.append(("45% на 1000 ставках: верхняя граница ниже порога", "теряете" in t3))
    ok.append(("ничьи не входят в статистику", "ничьих 2" in stats(list("wwlld" + "d"), 0.92) or "ничьих 2" in stats(list("wl") + ["d", "d"], 0.92)))
    ok.append(("серии считаются верно", longest(list("wwwlww"), "w") == 3))
    lo, hi = wilson(50, 100)
    ok.append(("интервал для 50/100 около 40%…60%", 0.39 < lo < 0.41 and 0.59 < hi < 0.61))
    bad = 0
    for name, good in ok:
        print(("  ok   " if good else "  ОШИБКА  ") + name)
        bad += not good
    print(f"\nИтог: {len(ok) - bad}/{len(ok)} проверок пройдено")
    raise SystemExit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payout", type=float, default=0.92)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    rows = load()
    print("Журнал демо-ставок. w = выигрыш, l = проигрыш, d = ничья (можно с пометкой: 'w audcad otc up'), stat, undo, q.")
    print(stats([r[1] for r in rows], a.payout))
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        head, _, note = cmd.partition(" ")
        h = head.lower()
        if h in ("q", "quit", "exit"):
            break
        if h in ("w", "l", "d"):
            rows.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), h, note])
            save(rows)
            n = sum(r[1] in ("w", "l") for r in rows)
            print(f"  записано ({h}); всего ставок без ничьих: {n}")
        elif h == "undo" and rows:
            print("  удалено:", rows.pop())
            save(rows)
        elif h in ("stat", "стат"):
            print(stats([r[1] for r in rows], a.payout))
        else:
            print("  Команды: w, l, d (с пометкой), stat, undo, q")


if __name__ == "__main__":
    main()
