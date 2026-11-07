import ccxt

from xrocket import XRocketExchange, XRocketPublic


def make_exchange(name: str, mode: str = "paper", api_key: str = "", secret: str = "",
                  market: str = "spot"):
    if name == "xrocket":
        if mode == "paper":
            return XRocketPublic()
        if market == "futures":
            raise SystemExit("xRocket: реальные ордера пока только на споте (MARKET=spot).")
        if mode == "demo":
            raise SystemExit("xRocket: режима demo нет. Для тестов используйте MODE=testnet "
                             "(бот @xrocket_testnet_bot), для реальных денег MODE=live.")
        if not api_key:
            raise SystemExit("Для xRocket впишите Bearer-токен в API_KEY (.env). API_SECRET не нужен.")
        return XRocketExchange(api_key, testnet=(mode == "testnet"))
    opts = {"enableRateLimit": True}
    if market == "futures":
        opts["options"] = {"defaultType": "swap"}  # линейные USDT-контракты
    if mode in ("demo", "testnet", "live"):
        opts.update(apiKey=api_key, secret=secret)
    ex = getattr(ccxt, name)(opts)
    if mode in ("testnet", "demo"):
        try:
            if mode == "testnet":
                ex.set_sandbox_mode(True)
            else:  # демо-торговля биржи (например, Bybit Demo Trading)
                ex.enable_demo_trading(True)
        except (KeyError, TypeError, ccxt.NotSupported):
            raise SystemExit(f"{name}: режим {mode} в ccxt не поддерживается. "
                             "Используйте MODE=paper (виртуальные деньги) или live (реальные).")
    return ex
