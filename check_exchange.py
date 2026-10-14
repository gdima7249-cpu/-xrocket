"""Безопасная проверка подключения (ордера НЕ отправляются):  python check_exchange.py
Показывает: доступны ли рынки, цена, минимальный размер ордера по SYMBOL, баланс (если заданы ключи)."""
import config
from exchanges import make_exchange

mode = config.MODE if config.MODE != "paper" else "paper"
ex = make_exchange(config.EXCHANGE, mode, config.API_KEY, config.API_SECRET)
print(f"Биржа: {config.EXCHANGE}, режим: {mode}, пара: {config.SYMBOL}")
ex.load_markets()
m = ex.market(config.SYMBOL)
price = ex.fetch_ticker(config.SYMBOL)["last"]
lim = m.get("limits", {})
print(f"Цена: {price}")
print(f"Мин. количество: {lim.get('amount', {}).get('min')}  Мин. сумма ордера: {lim.get('cost', {}).get('min')}")
print(f"Комиссия maker/taker: {m.get('maker')}/{m.get('taker')}")
if mode != "paper" and config.API_KEY:
    bal = ex.fetch_balance()
    quote = config.SYMBOL.split("/")[1]
    print(f"Баланс {quote}: свободно {bal.get(quote, {}).get('free')}, всего {bal.get(quote, {}).get('total')}")
    print("Ключи работают.")
    if config.EXCHANGE == "xrocket" and not bal.get(quote, {}).get("free"):
        print("На ТОРГОВОМ балансе нет средств: переведите их с основного (Funding) на торговый (Trading).")
else:
    print("Ключи не заданы — проверены только публичные данные.")
