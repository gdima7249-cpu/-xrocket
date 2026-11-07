"""Проверка ордеров на ТЕСТОВОЙ среде: покупает минимальную сумму и сразу продаёт обратно.
Запуск: python test_trade.py   (работает только при MODE=testnet или MODE=demo)"""
import config
from exchanges import make_exchange

if config.MODE not in ("testnet", "demo"):
    raise SystemExit("Только для MODE=testnet или MODE=demo (тестовые деньги). Сейчас: " + config.MODE)

ex = make_exchange(config.EXCHANGE, config.MODE, config.API_KEY, config.API_SECRET, config.MARKET)
ex.load_markets()
sym = config.SYMBOL
base, quote = sym.split(":")[0].split("/")
lim = ex.market(sym).get("limits", {})
min_cost = (lim.get("cost") or {}).get("min") or 0
min_amt = (lim.get("amount") or {}).get("min") or 0
price = ex.fetch_ticker(sym)["last"]
qty = float(ex.amount_to_precision(sym, max(min_amt * 1.5, min_cost * 1.5 / price)))


def show(title):
    b = ex.fetch_balance()
    print(f"{title}: {quote} {b[quote]['free']}, {base} {b[base]['free']}")


print(f"Тест на {config.EXCHANGE} ({config.MODE}): купить {qty} {base} по цене ~{price} (≈{qty * price:.4f} {quote}), затем продать.")
if input("Введите YES для продолжения: ") != "YES":
    raise SystemExit("Отменено")
show("До")
buy = ex.create_order(sym, "market", "buy", qty, price)
print("Покупка:", {k: buy.get(k) for k in ("id", "filled", "average", "fill_checked")})
show("После покупки")
got = buy.get("filled") if buy.get("fill_checked") else qty
free_base = ex.fetch_balance()[base]["free"]
sell_qty = float(ex.amount_to_precision(sym, min(got or 0, free_base)))
if sell_qty > 0:
    sell = ex.create_market_sell_order(sym, sell_qty)
    print("Продажа:", {k: sell.get(k) for k in ("id", "filled", "average", "fill_checked")})
else:
    print("Покупка не исполнилась — продавать нечего.")
show("После продажи")
