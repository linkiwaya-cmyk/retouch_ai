import requests, json, time, io, os
from dotenv import load_dotenv
from PIL import Image, ImageOps
import pillow_heif

load_dotenv()

RETOUCH_TOKEN = os.getenv("RETOUCH4ME_TOKEN")
BASE_URL = "https://cf-retoucher.retouch4.me/api/v1"

# ══════════════════════════════════════════════════════════════════════════════
# КОНЦЕПЦИЯ: локальная ретушь лица и кожи
#
# УБРАНО НАВСЕГДА:
# - Skin Tone Alpha2 (глобальный цветовой сдвиг всего кадра)
# - Dodge Burn Alpha2 / warmth (глобальный тёплый сдвиг)
# - _add_warmth() (глобальная RGB коррекция)
# - _fix_colors() (глобальная цветокоррекция)
# - любые PIL-постобработки цвета
#
# ОСТАВЛЕНО:
# - Fabric (текстура кожи)
# - Eye Vessels / Eye Brilliance (глаза)
# - White Teeth (зубы)
# - Dodge Burn Alpha1 only (локальная светотень лица)
# - Skin Tone Alpha1 only (тон кожи, не всего кадра)
# - Portrait Volumes (объём лица)
# - Heal (дефекты кожи)
# ══════════════════════════════════════════════════════════════════════════════

# Дефолтный пресет — Natural Retouch
PRESET = {
    "mode": "professional",
    "tasks": [
        {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.28},
        {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 0.8},
        {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.35},
        {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.2,  "Alpha2": 0.2},
        {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 0.85, "Alpha2": 0.0},
        {"Plugin": "Portrait Volumes", "Scale": 0, "Alpha1": 0.25},
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
    Отправляет оригинал в API без resize.
    Никакой постобработки цвета — возвращаем результат API как есть.
    """
    original_img = _open_image(image_bytes)
    original_size = original_img.size

    # Отправляем оригинал напрямую — без resize
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

    # Открываем результат
    try:
        result_img = Image.open(io.BytesIO(result_bytes)).convert('RGB')
    except Exception:
        return result_bytes

    # Проверяем размер — если API вернул меньше, ресайзим обратно
    if result_img.size != original_size:
        result_img = result_img.resize(original_size, Image.LANCZOS)

    # Никакой постобработки цвета — возвращаем как есть
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
