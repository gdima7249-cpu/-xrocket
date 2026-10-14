"""Безопасная проверка подключения (ордера НЕ отправляются):  python check_exchange.py
Показывает: доступны ли рынки, цена, минимальный размер ордера по SYMBOL, баланс (если заданы ключи)."""
import ccxt

import config
from exchanges import make_exchange

mode = config.MODE if config.MODE != "paper" else "paper"
ex = make_exchange(config.EXCHANGE, mode, config.API_KEY, config.API_SECRET)
print(f"Биржа: {config.EXCHANGE}, режим: {mode}, пара: {config.SYMBOL}")
ex.load_markets()
try:
    m = ex.market(config.SYMBOL)
except ccxt.ExchangeError as e:
    if config.EXCHANGE == "xrocket" and "symbol_not_found" in str(e):
        print(f"\nПары {config.SYMBOL} на этой среде нет. Доступные пары ({mode}):")
        for s in ex.list_symbols():
            print(f"  {s['symbol'].replace('-', '/'):<14} торговля: {'да' if s.get('enableTrading') else 'нет'}  "
                  f"мин. кол-во: {s.get('baseMinSize')}  мин. сумма: {s.get('quoteMinSize')}")
        raise SystemExit("\nВыберите пару из списка и впишите в .env, например SYMBOL=TON/USDT")
    raise
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
    if config.EXCHANGE == "xrocket" and not bal[quote]["free"]:
        print("На ТОРГОВОМ балансе нет средств: переведите их с основного (Funding) на торговый (Trading).")
else:
    print("Ключи не заданы — проверены только публичные данные.")
