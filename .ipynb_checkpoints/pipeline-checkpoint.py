import requests, json, time, io, os
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageOps
import pillow_heif
import numpy as np

load_dotenv()

RETOUCH_TOKEN = os.getenv("RETOUCH4ME_TOKEN")
BASE_URL = "https://cf-retoucher.retouch4.me/api/v1"

# ══════════════════════════════════════════════════════════════════════════════
# PRESET v4 — Natural Expensive Beauty Retouch / Dodge & Burn
# ══════════════════════════════════════════════════════════════════════════════

PRESET = {
    "mode": "professional",
    "tasks": [
        # Heal: ОТКЛЮЧЁН — родинки/веснушки/beauty marks сохраняются
        # {"Plugin": "Heal", "Scale": 0, "Alpha1": 0.0},

        # Fabric: сглаживание текстуры — поры видны
        {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.30},

        # Eye Vessels: убирает красные прожилки глаз
        {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 1.0},

        # Eye Brilliance: блеск глаз
        {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.5},

        # White Teeth: отбеливание зубов
        {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.25, "Alpha2": 0.25},

        # Dodge & Burn: главный инструмент выравнивания
        {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 0.95, "Alpha2": 0.25},

        # Skin Tone: цвет кожи, убирает green shift
        {"Plugin": "Skin Tone",        "Scale": 0, "Alpha1": 1.0,  "Alpha2": 0.55},

        # Portrait Volumes: объём лица
        {"Plugin": "Portrait Volumes", "Scale": 0, "Alpha1": 0.40},
    ]
}

# Максимальная сторона для отправки в API (Retouch4me лимит)
# Если фото больше — уменьшаем для API, потом апскейлим результат обратно
API_MAX_SIZE = 4096


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _open_image(image_bytes: bytes) -> Image.Image:
    """
    Открывает изображение с поддержкой HEIC/HEIF.
    Применяет EXIF-ориентацию чтобы не было перевёртышей.
    Конвертирует в RGB.
    """
    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(image_bytes))

    # Применяем EXIF-ориентацию (важно для iPhone фото)
    img = ImageOps.exif_transpose(img)

    # Конвертируем в RGB
    if img.mode not in ('RGB',):
        img = img.convert('RGB')

    return img


def _save_jpeg(img: Image.Image, quality: int = 97) -> bytes:
    """
    Сохраняет в JPEG с максимальным качеством.
    subsampling=0 — сохраняет цветовую детализацию (4:4:4 вместо 4:2:0)
    optimize=False — не тратим время на оптимизацию
    """
    buf = io.BytesIO()
    img.save(
        buf,
        format='JPEG',
        quality=quality,
        subsampling=0,    # 4:4:4 — максимальное качество цвета
        optimize=False,
    )
    return buf.getvalue()


def _resize_for_api(img: Image.Image, max_size: int) -> tuple[Image.Image, float]:
    """
    Уменьшает изображение для отправки в API если нужно.
    Возвращает (уменьшенное_изображение, scale_factor).
    scale_factor = 1.0 если resize не нужен.
    """
    w, h = img.size
    max_side = max(w, h)

    if max_side <= max_size:
        return img, 1.0

    scale = max_size / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale


def _upscale_to_original(
    retouched: Image.Image,
    original_size: tuple[int, int]
) -> Image.Image:
    """
    Апскейлит результат API обратно до оригинального размера.
    Используется только если мы уменьшали фото перед отправкой.
    """
    if retouched.size == original_size:
        return retouched
    return retouched.resize(original_size, Image.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing: цветокоррекция
# ══════════════════════════════════════════════════════════════════════════════

def _fix_colors(img: Image.Image) -> Image.Image:
    """
    Лёгкая цветокоррекция после Retouch4me API:
    - убирает green tint
    - восстанавливает насыщенность (+8%)
    - лёгкий warmth (+3% red)
    """
    arr = np.array(img, dtype=np.float32)

    # Нейтрализация green shift
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()

    if g_mean > r_mean + 2:
        green_excess = (g_mean - r_mean) * 0.4
        arr[:, :, 1] = np.clip(arr[:, :, 1] - green_excess, 0, 255)

    # Лёгкий warmth
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.03, 0, 255)

    img_fixed = Image.fromarray(arr.astype(np.uint8))

    # Восстанавливаем насыщенность
    img_fixed = ImageEnhance.Color(img_fixed).enhance(1.08)

    return img_fixed


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_image(image_bytes: bytes, filename: str) -> bytes:
    """
    Полный pipeline с сохранением оригинального разрешения:

    1. Открываем оригинал → запоминаем размер
    2. Если > API_MAX_SIZE → уменьшаем для API
    3. Отправляем в Retouch4me
    4. Получаем результат
    5. Если уменьшали → апскейлим результат обратно до оригинала
    6. Цветокоррекция
    7. Сохраняем в JPEG quality=97, subsampling=0
    """
    # ── Шаг 1: открываем оригинал ─────────────────────────────
    original_img = _open_image(image_bytes)
    original_size = original_img.size  # (width, height)
    original_w, original_h = original_size

    # ── Шаг 2: подготовка для API ─────────────────────────────
    api_img, scale = _resize_for_api(original_img, API_MAX_SIZE)
    was_resized = scale < 1.0

    # Конвертируем в bytes для API
    api_bytes = _save_jpeg(api_img, quality=97)
    api_filename = filename.rsplit('.', 1)[0] + '.jpg'

    # ── Шаг 3: отправляем в Retouch4me API ───────────────────
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

    # ── Шаг 4: ждём результат ─────────────────────────────────
    for _ in range(120):  # до 4 минут
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
        raise Exception("Timeout: Retouch4me не ответил за 4 минуты")

    result_bytes = requests.get(
        f"{BASE_URL}/retoucher/getFile/{task_id}",
        timeout=60,
    ).content

    # ── Шаг 5: открываем результат API ───────────────────────
    try:
        result_img = Image.open(io.BytesIO(result_bytes)).convert('RGB')
    except Exception:
        return result_bytes  # если не можем открыть — отдаём как есть

    # ── Шаг 6: апскейл обратно если уменьшали ────────────────
    if was_resized:
        result_img = _upscale_to_original(result_img, original_size)

    # ── Шаг 7: цветокоррекция ─────────────────────────────────
    try:
        result_img = _fix_colors(result_img)
    except Exception:
        pass  # если упало — продолжаем без цветокоррекции

    # ── Шаг 8: сохраняем с максимальным качеством ─────────────
    output = _save_jpeg(result_img, quality=97)

    return output


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