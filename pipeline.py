import requests, json, time, io, os
from dotenv import load_dotenv
from PIL import Image, ImageOps
import pillow_heif

load_dotenv()

RETOUCH_TOKEN = os.getenv("RETOUCH4ME_TOKEN")
BASE_URL = "https://cf-retoucher.retouch4.me/api/v1"

# ══════════════════════════════════════════════════════════════════════════════
# PRESET v5 — Natural Dodge & Burn / Professional Clean Retouch
#
# Философия: минимальное вмешательство, максимальная натуральность
#
# УДАЛЕНО:
# - Skin Tone Alpha2 (не поддерживается по документации API)
# - Dodge Burn warmth (Alpha2) — вызывал color imbalance
# - _fix_colors() — глобальная RGB коррекция убивала губы и нейтралы
# - Heal — удалял родинки
#
# ОСТАВЛЕНО:
# - Fabric: лёгкое сглаживание текстуры
# - Eye Vessels + Brilliance: глаза
# - White Teeth: минимально
# - Dodge Burn: основная работа, без warmth
# - Skin Tone: только Alpha1, мягко
# - Portrait Volumes: минимально
# ══════════════════════════════════════════════════════════════════════════════

PRESET = {
    "mode": "professional",
    "tasks": [
        # Fabric: лёгкое сглаживание — поры и текстура видны
        {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.25},

        # Eye Vessels: убирает красные прожилки
        {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 0.9},

        # Eye Brilliance: блеск глаз
        {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.4},

        # White Teeth: минимально
        {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.2, "Alpha2": 0.2},

        # Dodge & Burn: основной инструмент выравнивания
        # Alpha1: сила, Alpha2: warmth = 0 (без теплого сдвига)
        {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 0.85, "Alpha2": 0.0},

        # Skin Tone: только Alpha1 (Alpha2 не поддерживается!)
        # 0.6 — мягкое выравнивание без green/gray shift
        {"Plugin": "Skin Tone",        "Scale": 0, "Alpha1": 0.6},

        # Portrait Volumes: минимально — не меняем форму лица
        {"Plugin": "Portrait Volumes", "Scale": 0, "Alpha1": 0.25},
    ]
}

API_MAX_SIZE = 4096  # максимальная сторона для API


def _open_image(image_bytes: bytes) -> Image.Image:
    """Открывает изображение с поддержкой HEIC. Применяет EXIF-ориентацию."""
    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ('RGB',):
        img = img.convert('RGB')
    return img


def _save_jpeg(img: Image.Image, quality: int = 96) -> bytes:
    """Сохраняет в JPEG с высоким качеством. subsampling=0 = 4:4:4."""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, subsampling=0, optimize=False)
    return buf.getvalue()


def _resize_for_api(img: Image.Image, max_size: int):
    """Уменьшает если нужно. Возвращает (img, scale)."""
    w, h = img.size
    max_side = max(w, h)
    if max_side <= max_size:
        return img, 1.0
    scale = max_size / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS), scale


def process_image(image_bytes: bytes, filename: str) -> bytes:
    """
    Pipeline v5:
    1. Открываем оригинал → запоминаем размер
    2. Если > 4096px → уменьшаем для API
    3. Отправляем в Retouch4me
    4. Получаем результат
    5. Апскейл обратно если уменьшали
    6. Сохраняем quality=96 без лишней цветокоррекции
    
    НЕТ глобальной RGB коррекции — она ломала губы и нейтралы.
    """
    # Открываем оригинал
    original_img = _open_image(image_bytes)
    original_size = original_img.size

    # Подготовка для API
    api_img, scale = _resize_for_api(original_img, API_MAX_SIZE)
    was_resized = scale < 1.0
    api_bytes = _save_jpeg(api_img, quality=95)
    api_filename = filename.rsplit('.', 1)[0] + '.jpg'

    # Отправляем в Retouch4me API
    resp = requests.post(
        f"{BASE_URL}/retoucher/start",
        files={"file": (api_filename, api_bytes, "image/jpeg")},
        data={"token": RETOUCH_TOKEN, "payload": json.dumps(PRESET)},
        timeout=60,
    )
    data = resp.json()
    if data.get("status") != 200:
        raise Exception(f"Retouch4me error: {data}")

    task_id = data["id"]

    # Ждём результат
    for _ in range(120):
        time.sleep(2)
        s = requests.get(
            f"{BASE_URL}/retoucher/status/{task_id}",
            timeout=15,
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

    # Апскейл обратно если уменьшали
    if was_resized:
        result_img = result_img.resize(original_size, Image.LANCZOS)

    # Сохраняем — без лишней цветокоррекции
    return _save_jpeg(result_img, quality=96)


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
