"""Подсказки для РУЧНЫХ ставок «вверх» (бинарный опцион на демо-счёте) по правилу отката после резкого падения.

Бот ничего не покупает и не подключается к брокеру: он следит за реальными ценами Binance (публично, ключ не нужен) и, когда монета
резко упала, пищит и пишет: «ставка ВВЕРХ на 60 минут». Ставку вы делаете сами на демо-счёте брокера по НЕ-OTC паре
(у OTC-пар цену рисует сам брокер — к ним сигнал не относится).

Правила (те же, что в rebound_paper.py, проверены на истории):
  B (основное): падение ≥3% за 15 минут → ВВЕРХ на 60 минут. По истории цена через час была выше примерно в 60% случаев.
  A (слабее):   падение ≥2% за 5 минут  → ВВЕРХ на 60 минут.
Порог безубыточности бинарной ставки: выплата 92% → 52.1%, 85% → 54.1%, 80% → 55.6%.

Чтобы понять, работает ли это у ВАШЕГО брокера, записывайте каждую ставку в журнал (бот спросит итог через час, можно пропустить):
rebound_alerts.csv. Для вывода нужно 50–100 ставок.

  python rebound_alerts.py                           # монеты по умолчанию, правило B
  python rebound_alerts.py --coins BTC,ETH,SOL,XRP   # только пары, которые есть у вашего брокера
  python rebound_alerts.py --rules A,B               # оба правила (на одном обвале может прийти два сигнала по одной монете)
  python rebound_alerts.py --report
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta

import rebound_paper as RP

LOG = "rebound_alerts.csv"


def beep():
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1200, 250)
            time.sleep(0.1)
    except Exception:  # noqa: BLE001
        sys.stdout.write("\a\a\a")
        sys.stdout.flush()


def report():
    if not os.path.exists(LOG):
        return "Журнал пуст."
    with open(LOG, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("итог") in ("в", "п")]
    if not rows:
        return "В журнале нет ставок с итогом."
    n, w = len(rows), sum(r["итог"] == "в" for r in rows)
    p = w / n
    se = (p * (1 - p) / n) ** 0.5
    return (f"Ставок с итогом: {n}, выиграно {w} ({100 * p:.1f}% ± {200 * se:.1f}%). "
            f"Порог безубыточности: 52.1% при выплате 92%, 54.1% при 85%, 55.6% при 80%. Для вывода нужно 50–100 ставок.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coins", default="BTC,ETH,SOL,XRP,DOGE,BNB,ADA,AVAX,LINK,LTC,DOT,TRX")
    ap.add_argument("--rules", default="B")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    rules = set(a.rules.split(","))
    RP.RULES = {k: v for k, v in RP.RULES.items() if k in rules}
    RP.STATE = "rebound_alerts_state.json"
    RP._write = lambda rec: None                                     # журнал бумажного бота здесь не нужен
    coins = a.coins.split(",")
    pending = []                                                     # (время итога, монета, правило, цена)

    def on_msg(m):
        if "СИГНАЛ" not in m:
            return
        parts = m.split()
        coin, rule = parts[1], parts[2].strip("[]:")
        until = datetime.now() + timedelta(minutes=60)
        print("\n" + "=" * 70)
        print(f"🔔 {datetime.now():%H:%M}  СИГНАЛ [{rule}]: {coin} резко упала → ставка ВВЕРХ на 60 минут (до {until:%H:%M})")
        print(f"   {m.strip()}")
        print("=" * 70 + "\n")
        beep()
        pending.append((time.time() + 3600, coin, rule, until))

    st = RP.load_state()
    print(f"Слежу за: {', '.join(coins)}; правила {', '.join(sorted(rules))}. Ничего не покупаю. Ctrl+C — стоп.")
    print("При сигнале будет звук и надпись. Ставьте ВВЕРХ только по НЕ-OTC паре, срок 60 минут.\n" + report())
    while True:
        now = int(time.time())
        for coin in coins:
            try:
                RP.step(st, coin, RP.fetch(coin + "USDT"), now, on_msg)
            except Exception as e:  # noqa: BLE001
                print(f"  {coin}: ошибка загрузки ({type(e).__name__}: {str(e)[:60]})")
        RP.save_state(st)
        for item in [x for x in pending if x[0] <= time.time()]:
            pending.remove(item)
            _, coin, rule, until = item
            beep()
            ans = input(f"Ставка ВВЕРХ по {coin} [{rule}] до {until:%H:%M} закончилась. Итог? (в — выиграл, п — проиграл, "
                        f"Enter — не ставил): ").strip().lower()[:1]
            new = not os.path.exists(LOG)
            with open(LOG, "a", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(["время", "монета", "правило", "итог"])
                wr.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), coin, rule, ans or "-"])
            print(report())
        time.sleep(30)


if __name__ == "__main__":
    main()
