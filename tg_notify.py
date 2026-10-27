"""Уведомления в Telegram от ботов (отчёты, остановка). Только отправка сообщений вам — бот ничего не читает и не делает.

Настройка (один раз):
  1. В Telegram откройте @BotFather → /newbot → придумайте имя → он пришлёт ТОКЕН (длинная строка с двоеточием).
  2. Впишите токен в .env:   TELEGRAM_BOT_TOKEN=...
  3. Откройте своего нового бота в Telegram и напишите ему любое сообщение (например «привет»).
  4. python3 tg_notify.py --setup     — найдёт номер вашего чата и допишет TELEGRAM_CHAT_ID в .env
  5. python3 tg_notify.py --test      — пришлёт пробное сообщение
Токен никому не показывайте: с ним можно писать от имени вашего бота.
"""
import argparse
import os

try:
    import config  # noqa: F401  (подгружает .env)
except Exception:  # noqa: BLE001
    pass

API = "https://api.telegram.org/bot{token}/{method}"


def _token():
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def enabled():
    return bool(_token() and os.getenv("TELEGRAM_CHAT_ID", "").strip())


BUTTONS = [["💰 Баланс", "📈 Текущие ставки"], ["📜 История", "📊 Разбивка"]]
COMMANDS = {"💰 баланс": "balance", "/balance": "balance", "баланс": "balance",
            "📈 текущие ставки": "open", "/open": "open", "текущие ставки": "open", "ставки": "open",
            "📜 история": "history", "/history": "history", "история": "history",
            "📊 разбивка": "stats", "/stats": "stats", "разбивка": "stats",
            "/start": "help", "/help": "help", "помощь": "help"}


def send(text, keyboard=True):
    """Отправить сообщение (с кнопками команд). Если Telegram не настроен или недоступен — молча ничего не делает."""
    if not enabled():
        return False
    try:
        import json
        import requests
        data = {"chat_id": os.getenv("TELEGRAM_CHAT_ID").strip(), "text": text[:4000]}
        if keyboard:
            data["reply_markup"] = json.dumps({"keyboard": [[{"text": b} for b in row] for row in BUTTONS], "resize_keyboard": True})
        r = requests.post(API.format(token=_token(), method="sendMessage"), data=data, timeout=15)
        return r.ok
    except Exception:  # noqa: BLE001
        return False


class Commands:
    """Читает новые сообщения боту и отвечает на команды. Отвечает ТОЛЬКО владельцу (TELEGRAM_CHAT_ID), остальных игнорирует.
    handlers: {"balance": функция() -> текст, "open": ..., "history": ...}."""

    def __init__(self, handlers, get=None):
        self.handlers, self.offset, self._get = handlers, None, get
        self._skip_old()

    def _updates(self, timeout=0):
        if self._get:
            return self._get(self.offset)
        import requests
        params = {"timeout": timeout}
        if self.offset is not None:
            params["offset"] = self.offset
        r = requests.get(API.format(token=_token(), method="getUpdates"), params=params, timeout=timeout + 15).json()
        return r.get("result", []) if r.get("ok") else []

    def _skip_old(self):
        """Старые команды (отправленные, пока бот был выключен) не выполняем."""
        if not enabled() and not self._get:
            return
        try:
            ups = self._updates()
            if ups:
                self.offset = ups[-1]["update_id"] + 1
        except Exception:  # noqa: BLE001
            pass

    def poll(self, reply=None):
        if not enabled() and not self._get:
            return 0
        reply = reply or send
        try:
            ups = self._updates()
        except Exception:  # noqa: BLE001
            return 0
        n = 0
        owner = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        for u in ups:
            self.offset = u["update_id"] + 1
            m = u.get("message") or {}
            if str((m.get("chat") or {}).get("id")) != owner:
                continue                                             # чужие сообщения игнорируем
            cmd = COMMANDS.get((m.get("text") or "").strip().lower())
            if cmd == "help" or cmd is None:
                reply("Команды: 💰 Баланс, 📈 Текущие ставки, 📜 История, 📊 Разбивка (кнопки внизу).")
            else:
                try:
                    reply(self.handlers[cmd]())
                except Exception as e:  # noqa: BLE001
                    reply(f"Не получилось: {type(e).__name__}: {str(e)[:150]}")
            n += 1
        return n


def setup(env_path=".env"):
    import requests
    if not _token():
        raise SystemExit("Сначала впишите TELEGRAM_BOT_TOKEN=... в .env (токен от @BotFather).")
    r = requests.get(API.format(token=_token(), method="getUpdates"), timeout=15).json()
    if not r.get("ok"):
        raise SystemExit(f"Telegram ответил ошибкой: {r.get('description')} (проверьте токен)")
    chats = {}
    for u in r.get("result", []):
        m = u.get("message") or u.get("edited_message") or {}
        c = m.get("chat") or {}
        if c.get("id"):
            chats[c["id"]] = c.get("first_name") or c.get("title") or ""
    if not chats:
        raise SystemExit("Сообщений боту не найдено. Откройте своего бота в Telegram, напишите ему «привет» и запустите снова.")
    chat_id, name = list(chats.items())[-1]
    lines = open(env_path, encoding="utf-8").read().splitlines() if os.path.exists(env_path) else []
    lines = [ln for ln in lines if not ln.startswith("TELEGRAM_CHAT_ID=")] + [f"TELEGRAM_CHAT_ID={chat_id}"]
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.environ["TELEGRAM_CHAT_ID"] = str(chat_id)
    print(f"Чат найден ({name}), TELEGRAM_CHAT_ID записан в {env_path}.")
    print("Пробное сообщение:", "отправлено" if send("✅ Уведомления от ботов настроены.") else "не отправилось")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.setup:
        setup()
    elif a.test:
        print("отправлено" if send("Проверка: уведомления работают.") else "не отправилось (проверьте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID в .env)")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
