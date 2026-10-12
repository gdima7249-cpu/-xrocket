import ccxt

from xrocket import XRocketPublic


def make_exchange(name: str, mode: str = "paper", api_key: str = "", secret: str = ""):
    if name == "xrocket":
        if mode != "paper":
            raise SystemExit("Для xrocket пока реализован только MODE=paper "
                             "(ордера требуют документации Place order и Balances).")
        return XRocketPublic()
    opts = {"enableRateLimit": True}
    if mode in ("demo", "testnet", "live"):
        opts.update(apiKey=api_key, secret=secret)
    ex = getattr(ccxt, name)(opts)
    if mode == "testnet":
        ex.set_sandbox_mode(True)
    elif mode == "demo":  # демо-торговля биржи (например, Bybit Demo Trading)
        if not hasattr(ex, "enable_demo_trading"):
            raise SystemExit(f"{name}: демо-режим в ccxt не поддерживается, используйте testnet")
        ex.enable_demo_trading(True)
    return ex
