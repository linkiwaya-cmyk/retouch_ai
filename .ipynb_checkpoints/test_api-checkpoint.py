import requests
import json
import time
import os
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

TOKEN = os.getenv("RETOUCH4ME_TOKEN")

BASE = "https://cf-retoucher.retouch4.me/api/v1"

# Проверка токена
if not TOKEN:
    print("❌ RETOUCH4ME_TOKEN не найден в .env")
    exit()

print("✅ TOKEN найден")

# Проверка баланса
print("\n🔍 Проверяю баланс...")

r = requests.post(
    "https://3dlutcreator.com/api/retouch/v1/balance",
    data={
        "retouchtoken": TOKEN,
        "modes[]": "professional"
    }
)

print("Баланс:", r.json())

# Пресет ретуши
PAYLOAD = {
    "mode": "professional",
    "tasks": [
        {"Plugin": "Heal",             "Scale": 0, "Alpha1": 1.0},
        {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.39},
        {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 1.0},
        {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.5},
        {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.25, "Alpha2": 0.25},
        {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 1.0,  "Alpha2": 0.2},
        {"Plugin": "Skin Tone",        "Scale": 0, "Alpha1": 1.0,  "Alpha2": 1.0},
        {"Plugin": "Portrait Volumes", "Scale": 0, "Alpha1": 0.5}
    ]
}

# Проверяем наличие test.jpg
if not os.path.exists("test.jpg"):
    print("\n❌ Файл test.jpg не найден")
    print("Загрузи фото в папку /workspace/retouch_ai")
    exit()

print("\n📤 Отправляю фото в Retouch4me...")

# Отправка фото
with open("test.jpg", "rb") as f:

    r = requests.post(
        f"{BASE}/retoucher/start",
        files={
            "file": ("test.jpg", f, "image/jpeg")
        },
        data={
            "token": TOKEN,
            "payload": json.dumps(PAYLOAD)
        }
    )

data = r.json()

print("\nОтвет start:")
print(data)

# Проверка ошибки
if data.get("status") != 200:
    print("\n❌ ОШИБКА ПРИ СТАРТЕ")
    exit()

task_id = data["id"]

print(f"\n🆔 task_id: {task_id}")

# Polling статуса
print("\n⏳ Жду обработку...\n")

for i in range(60):

    time.sleep(2)

    s = requests.get(
        f"{BASE}/retoucher/status/{task_id}"
    ).json()

    print(f"[{i*2}с] state={s.get('state')} progress={s.get('progress',0)}%")

    # Готово
    if s.get("state") == "completed":

        print("\n📥 Скачиваю результат...")

        result = requests.get(
            f"{BASE}/retoucher/getFile/{task_id}"
        )

        with open("result.jpg", "wb") as out:
            out.write(result.content)

        print("\n✅ ГОТОВО → result.jpg")
        break

    # Ошибка
    elif s.get("state") == "failed":

        print("\n❌ FAILED")
        print(s)
        break