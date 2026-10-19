"""Показывает ваши счета Deriv Options (номер, тип, валюта, баланс) — чтобы узнать DERIV_ACCOUNT_ID демо-счёта.
Токен берётся из .env (DERIV_TOKEN, DERIV_APP_ID), на экран не выводится. Только чтение, ничего не покупает.

  python deriv_accounts.py
"""
import json
import os
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv(encoding="utf-8-sig")
except Exception:  # noqa: BLE001
    pass

URL = "https://api.derivws.com/trading/v1/options/accounts"


def fetch(token, app_id):
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token}", "Deriv-App-ID": app_id,
                                               "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    token = os.getenv("DERIV_TOKEN", "").strip().strip("\"'").strip()
    app_id = os.getenv("DERIV_APP_ID", "").strip().strip("\"'").strip()
    print(f"Токен: длина {len(token)}, начинается с «{token[:4]}…» (сам токен не показываем); App ID: длина {len(app_id)}.")
    print("Подсказка: токен — длинная строка (обычно 30+ символов) без пробелов и кавычек; App ID — короче.")
    if not token or not app_id:
        raise SystemExit("Заполните DERIV_TOKEN и DERIV_APP_ID в файле .env")
    try:
        data = fetch(token, app_id)
    except Exception as e:  # noqa: BLE001
        body = getattr(e, "read", lambda: b"")().decode(errors="ignore")[:300]
        raise SystemExit(f"Запрос не удался: {type(e).__name__} {body}")
    items = data.get("data", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = items.get("accounts", [items])
    for a in items if isinstance(items, list) else []:
        print({k: v for k, v in a.items() if "token" not in k.lower()})
    print("\nДемо-счёт — тот, где тип demo (или номер начинается с VRT/DOT…). Его номер пойдёт в DERIV_ACCOUNT_ID.")


if __name__ == "__main__":
    main()
