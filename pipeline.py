import requests, json, time, io, os
from dotenv import load_dotenv
from PIL import Image, ImageOps
import pillow_heif

load_dotenv()

RETOUCH_TOKEN = os.getenv("RETOUCH4ME_TOKEN")
BASE_URL = "https://cf-retoucher.retouch4.me/api/v1"

# ══════════════════════════════════════════════════════════════════════════════
# ПРАВИЛО: бот делает ТОЛЬКО ретушь лица и кожи.
# Фотография после обработки должна выглядеть как исходник —
# тот же контраст, цвет, экспозиция, тон, насыщенность, фон, волосы, тело.
# Разница только в более чистой коже лица.
#
# ЗАПРЕЩЕНО:
# - менять контраст, цвет, насыщенность, экспозицию фото
# - любые PIL постобработки цвета/яркости/контраста
# - sharpening (он меняет восприятие всего кадра)
# - Dodge Burn с Alpha > 0.3 (агрессивно меняет светотень лица)
# - Skin Tone Alpha2 (глобальный цветовой сдвиг)
#
# РАЗРЕШЕНО:
# - Heal (убирает дефекты кожи — прыщи, пятна)
# - Fabric (текстура кожи, только мягко Alpha <= 0.25)
# - Eye Vessels (убирает красноту глаз)
# - Dodge Burn Alpha1 только очень мягко (локальная светотень лица)
# ══════════════════════════════════════════════════════════════════════════════

# Дефолтный пресет — используется если preset=None
PRESET = {
    "mode": "professional",
    "tasks": [
        {"Plugin": "Heal",        "Scale": 0, "Alpha1": 0.5},
        {"Plugin": "Fabric",      "Scale": 0, "Alpha1": 0.20},
        {"Plugin": "Eye Vessels", "Scale": 0, "Alpha1": 0.6},
        {"Plugin": "Dodge Burn",  "Scale": 1, "Alpha1": 0.20, "Alpha2": 0.0},
    ]
}


def _open_image(image_bytes: bytes) -> Image.Image:
    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def _save_jpeg(img: Image.Image, quality: int = 100) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, subsampling=0, optimize=False)
    return buf.getvalue()


def process_image(image_bytes: bytes, filename: str, preset: dict = None) -> bytes:
    """
    Отправляет оригинал в API без resize и без постобработки.
    Никаких изменений цвета, контраста, яркости после API.
    Только ретушь лица через Retouch4me.
    """
    original_img = _open_image(image_bytes)
    original_size = original_img.size

    api_bytes = _save_jpeg(original_img, quality=100)
    api_filename = filename.rsplit('.', 1)[0] + '.jpg'

    resp = requests.post(
        f"{BASE_URL}/retoucher/start",
        files={"file": (api_filename, api_bytes, "image/jpeg")},
        data={"token": RETOUCH_TOKEN, "payload": json.dumps(preset or PRESET)},
        timeout=60,
    )
    data = resp.json()
    if data.get("status") != 200:
        raise Exception(f"Retouch4me error: {data}")

    task_id = data["id"]

    for _ in range(150):
        time.sleep(2)
        s = requests.get(
            f"{BASE_URL}/retoucher/status/{task_id}",
            timeout=90,
        ).json()
        if s.get("state") == "completed":
            break
        if s.get("state") == "failed":
            raise Exception(f"Retouch4me failed: {s.get('reason')}")
    else:
        raise Exception("Timeout")

    result_bytes = requests.get(
        f"{BASE_URL}/retoucher/getFile/{task_id}",
        timeout=60,
    ).content

    try:
        result_img = Image.open(io.BytesIO(result_bytes)).convert('RGB')
    except Exception:
        return result_bytes

    # Если API вернул меньший размер — ресайзим обратно
    if result_img.size != original_size:
        result_img = result_img.resize(original_size, Image.LANCZOS)

    # Никакой постобработки — возвращаем как есть
    return _save_jpeg(result_img, quality=100)


def get_balance():
    try:
        r = requests.post(
            "https://3dlutcreator.com/api/retouch/v1/balance",
            data={"retouchtoken": RETOUCH_TOKEN, "modes[]": "professional"},
            timeout=10,
        )
        return r.json().get("remaining", {}).get("professional", 0)
    except Exception:
        return -1
