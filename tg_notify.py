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


def send(text):
    """Отправить сообщение. Если Telegram не настроен или недоступен — молча ничего не делает (бот не должен падать)."""
    if not enabled():
        return False
    try:
        import requests
        r = requests.post(API.format(token=_token(), method="sendMessage"),
                          data={"chat_id": os.getenv("TELEGRAM_CHAT_ID").strip(), "text": text[:4000]}, timeout=15)
        return r.ok
    except Exception:  # noqa: BLE001
        return False


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
