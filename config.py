import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Params:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_max_entry: float = 70.0     # не входить, если рынок перегрет
    stop_loss_pct: float = 0.03     # стоп 3%
    take_profit_pct: float = 0.06   # тейк 6%
    risk_per_trade: float = 0.01    # рисковать 1% капитала на сделку
    max_position_pct: float = 0.5   # не более 50% капитала в одной позиции
    max_daily_loss_pct: float = 0.03  # стоп-торговля на день при -3%
    fee: float = 0.001              # комиссия 0.1% за сторону
    slippage: float = 0.0005        # проскальзывание 0.05%


MODE = os.getenv("MODE", "paper")
EXCHANGE = os.getenv("EXCHANGE", "xrocket")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
START_BALANCE = 1000.0  # для paper и бэктеста
