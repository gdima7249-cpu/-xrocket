import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Params:
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    rsi_max_entry: float = 70.0     # не открывать long, если рынок перегрет
    rsi_min_entry: float = 30.0     # не открывать short, если рынок перепродан
    stop_loss_pct: float = 0.03     # стоп 3%
    take_profit_pct: float = 0.06   # тейк 6%
    risk_per_trade: float = 0.01    # рисковать 1% капитала на сделку
    max_position_pct: float = 0.5   # не более 50% капитала в одной позиции
    max_daily_loss_pct: float = 0.03  # стоп-торговля на день при -3%
    fee: float = float(os.getenv("FEE", "0.001"))              # комиссия за сторону (xRocket ≈ 0.003)
    slippage: float = float(os.getenv("SLIPPAGE", "0.0005"))   # проскальзывание (xRocket ≈ 0.0025)


MODE = os.getenv("MODE", "paper")
STRATEGY = os.getenv("STRATEGY", "ema_rsi")    # ema_rsi | donchian (только спот)
DONCHIAN_IN = int(os.getenv("DONCHIAN_IN", "40"))    # вход: закрытие выше максимума за N свечей
DONCHIAN_OUT = int(os.getenv("DONCHIAN_OUT", "20"))  # выход: закрытие ниже минимума за M свечей
# Сканер просадок (scanner_bot.py): покупка резкого падения, продажа на отскоке
SCAN_DROP = float(os.getenv("SCAN_DROP", "0.06"))           # падение от максимума за LOOKBACK свечей, 0.06 = 6%
SCAN_LOOKBACK = int(os.getenv("SCAN_LOOKBACK", "24"))       # окно в свечах (1h) -> 24 часа
SCAN_CONFIRM = os.getenv("SCAN_CONFIRM", "1") == "1"        # покупать только после зелёной свечи (отскок начался)
SCAN_TAKE = float(os.getenv("SCAN_TAKE", "0.05"))           # тейк-профит
SCAN_STOP = float(os.getenv("SCAN_STOP", "0.05"))           # стоп-лосс
SCAN_MAX_HOLD_H = int(os.getenv("SCAN_MAX_HOLD_H", "48"))   # выйти по таймауту, часов
SCAN_MAX_POSITIONS = int(os.getenv("SCAN_MAX_POSITIONS", "5"))
SCAN_POSITION_PCT = float(os.getenv("SCAN_POSITION_PCT", "0.15"))  # доля капитала на позицию
MARKET = os.getenv("MARKET", "spot")          # spot (только покупки) | futures (long и short)
LEVERAGE = float(os.getenv("LEVERAGE", "2"))   # плечо для futures, от 1 до 5
EXCHANGE = os.getenv("EXCHANGE", "xrocket")
API_KEY = os.getenv("API_KEY") or os.getenv("XROCKET_TOKEN", "")
API_SECRET = os.getenv("API_SECRET", "")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Общий стоп-кран: при падении капитала до (1 - MAX_TOTAL_LOSS_PCT) от стартового
# бот продаёт позицию и полностью останавливается. 0.20 = стоп при 800 из 1000.
MAX_TOTAL_LOSS_PCT = float(os.getenv("MAX_TOTAL_LOSS_PCT", "0.20"))
# Либо абсолютный порог капитала в USDT (если задан — важнее процента)
STOP_EQUITY = float(os.getenv("STOP_EQUITY")) if os.getenv("STOP_EQUITY") else None
START_BALANCE = float(os.getenv("PAPER_BALANCE", "1000"))  # в котируемой валюте (USDT), для paper и бэктеста
