import requests, json, time, io, os
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageEnhance
import pillow_heif

load_dotenv()

RETOUCH_TOKEN = os.getenv("RETOUCH4ME_TOKEN")
BASE_URL = "https://cf-retoucher.retouch4.me/api/v1"

# ══════════════════════════════════════════════════════════════════════════════
# PRESET v6 — на основе официальной документации Retouch4me
#
# Изменения vs v5:
# - Skin Tone Alpha2 ВОЗВРАЩЁН (документация подтверждает что он есть!)
#   Alpha2 у Skin Tone = второй параметр выравнивания тона
# - Dodge Burn Alpha1 поднят до 0.95 — больше ретуши
# - Fabric 0.30 — чуть больше сглаживания
# - Добавлена лёгкая warmth коррекция через PIL после API
#   (только +красный, без изменения зелёного глобально)
# ══════════════════════════════════════════════════════════════════════════════

PRESET = {
    "mode": "professional",
    "tasks": [
        # Fabric: сглаживание текстуры
        {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.35},

        # Eye Vessels: убирает красные прожилки
        {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 0.9},

        # Eye Brilliance: блеск глаз
        {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.4},

        # White Teeth: минимально
        {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.2, "Alpha2": 0.2},

        # Dodge & Burn: основной инструмент — 95% силы
        # Alpha2 = warmth = 0.1 (минимальный тёплый оттенок)
        {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 0.95, "Alpha2": 0.0},  # warmth=0 — не трогаем губы

        # Skin Tone: выравнивание тона кожи
        # Alpha1 = сила выравнивания
        # Alpha2 = 0.3 (мягко, без green shift)
        {"Plugin": "Skin Tone",        "Scale": 0, "Alpha1": 0.75, "Alpha2": 0.0},  # Alpha2=0 — не меняем тон губ

        # Portrait Volumes: объём лица — мягко
        {"Plugin": "Portrait Volumes", "Scale": 0, "Alpha1": 0.35},
    ]
}

API_MAX_SIZE = 4096


def _open_image(image_bytes: bytes) -> Image.Image:
    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def _save_jpeg(img: Image.Image, quality: int = 96) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, subsampling=0, optimize=False)
    return buf.getvalue()


def _resize_for_api(img: Image.Image, max_size: int):
    w, h = img.size
    if max(w, h) <= max_size:
        return img, 1.0
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS), scale


def _add_warmth(img: Image.Image) -> Image.Image:
    """
    Минимальная warmth коррекция после API:
    - чуть поднимает красный канал (+2%) для натуральности губ
    - НЕ трогает зелёный и синий глобально
    - лёгкое поднятие насыщенности +5%
    """
    import numpy as np
    arr = np.array(img, dtype=np.float32)

    # Только лёгкий warmth — +2% красного
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.02, 0, 255)

    img_out = Image.fromarray(arr.astype(np.uint8))

    # Насыщенность +5%
    # Насыщенность не трогаем — меньше пересчётов пикселей
    # img_out = ImageEnhance.Color(img_out).enhance(1.05)

    return img_out


def process_image(image_bytes: bytes, filename: str, preset: dict = None) -> bytes:
    original_img = _open_image(image_bytes)
    original_size = original_img.size

    # НЕ делаем resize — API поддерживает до 250MP и 100MB
    # Отправляем оригинал напрямую
    api_bytes = _save_jpeg(original_img, quality=100)
    api_filename = filename.rsplit('.', 1)[0] + '.jpg'
    was_resized = False

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

    for _ in range(150):  # до 5 минут
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

    if was_resized:
        result_img = result_img.resize(original_size, Image.LANCZOS)

    # _add_warmth убрана — она меняла цвет губ глобально
    # Цвет губ теперь сохраняется как в оригинале

    # quality=100, subsampling=0 — максимальное качество на выходе
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
