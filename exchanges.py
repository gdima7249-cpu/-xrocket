import ccxt

from xrocket import XRocketPublic


def make_exchange(name: str, mode: str = "paper", api_key: str = "", secret: str = "",
                  market: str = "spot"):
    if name == "xrocket":
        if mode != "paper":
            raise SystemExit("Для xrocket пока реализован только MODE=paper "
                             "(ордера требуют документации Place order и Balances).")
        return XRocketPublic()
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
