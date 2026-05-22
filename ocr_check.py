"""
ocr_check.py — Retouch Lab
OCR-валидация скриншотов оплаты MBank / Visa.

Логика:
1. Скачиваем скриншот из Telegram
2. Пробуем извлечь: сумму, дату, имя
3. Сравниваем с ожидаемой суммой тарифа
4. Возвращаем вердикт + извлечённые данные для отображения админу

Используется pytesseract если установлен.
Если не установлен — работает в fallback режиме (только пересылка без OCR).
"""

import io
import re
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Пробуем импортировать OCR библиотеки
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    logger.info("OCR: pytesseract доступен")
except ImportError:
    OCR_AVAILABLE = False
    logger.info("OCR: pytesseract недоступен — работаем без OCR")


# ── Паттерны для извлечения данных из скриншотов MBank ───────────────────────

# Суммы: "1 300,00", "1300.00", "1 300 сом", "990 KGS" и т.д.
AMOUNT_PATTERNS = [
    r'(\d[\d\s]{2,6}[.,]\d{2})',          # 1 300,00 / 1300.00
    r'(\d+)\s*(?:сом|KGS|кгс|som)',        # 990 сом / 1300 KGS
    r'Сумма[:\s]+(\d[\d\s]*)',             # Сумма: 990
    r'Amount[:\s]+(\d[\d\s]*)',            # Amount: 990
    r'Итого[:\s]+(\d[\d\s]*)',             # Итого: 1300
]

# Дата: "20.05.2026", "2026-05-20", "20 мая 2026"
DATE_PATTERNS = [
    r'(\d{2}[./]\d{2}[./]\d{4})',          # 20.05.2026
    r'(\d{4}-\d{2}-\d{2})',                # 2026-05-20
    r'(\d{2}\s+\w+\s+\d{4})',              # 20 мая 2026
]

# Время: "14:35", "14:35:22"
TIME_PATTERNS = [
    r'(\d{2}:\d{2}(?::\d{2})?)',
]

# Имя получателя
NAME_PATTERNS = [
    r'(?:Получатель|Кому|To)[:\s]+([А-ЯA-Z][а-яa-z]+\s+[А-ЯA-Z]\.?)',
    r'([А-Я][а-я]+\s+[А-Я]\.\s*[А-Я]?\.?)',   # Алина А. или Иванов И.И.
]

# Transaction ID
TXN_PATTERNS = [
    r'(?:ID|Транзакция|Transaction)[:\s#]+(\w{6,})',
    r'#(\d{6,})',
]


# ══════════════════════════════════════════════════════════════════════════════
# Основная функция OCR
# ══════════════════════════════════════════════════════════════════════════════

def _clean_amount(raw: str) -> Optional[float]:
    """Конвертирует строку суммы в число. '1 300,00' → 1300.0"""
    try:
        cleaned = raw.replace(' ', '').replace(',', '.').replace('\xa0', '')
        return float(cleaned)
    except Exception:
        return None


def _extract_from_text(text: str, expected_amount: int) -> dict:
    """
    Извлекает данные из OCR-текста.
    Возвращает dict с найденными полями и вердиктом.
    """
    result = {
        "raw_text":       text[:500],   # первые 500 символов для отладки
        "amount_found":   None,
        "amount_match":   False,
        "date_found":     None,
        "date_fresh":     False,
        "time_found":     None,
        "name_found":     None,
        "txn_id":         None,
        "ocr_confidence": "low",
    }

    # ── Ищем сумму ────────────────────────────────────────────────────────────
    for pattern in AMOUNT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            amount = _clean_amount(match)
            if amount and amount > 100:  # игнорируем мелкие числа
                result["amount_found"] = amount
                # Допуск ±5 сом (для копеек/комиссии)
                if abs(amount - expected_amount) <= 5:
                    result["amount_match"] = True
                break
        if result["amount_found"]:
            break

    # ── Ищем дату ─────────────────────────────────────────────────────────────
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            result["date_found"] = matches[0]
            # Проверяем что дата не старше 24 часов
            try:
                for fmt in ('%d.%m.%Y', '%Y-%m-%d', '%d/%m/%Y'):
                    try:
                        dt = datetime.strptime(matches[0], fmt)
                        if datetime.now() - dt <= timedelta(days=1):
                            result["date_fresh"] = True
                        break
                    except Exception:
                        continue
            except Exception:
                pass
            break

    # ── Ищем время ────────────────────────────────────────────────────────────
    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            result["time_found"] = matches[0]
            break

    # ── Ищем имя ──────────────────────────────────────────────────────────────
    for pattern in NAME_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            result["name_found"] = matches[0].strip()
            break

    # ── Ищем Transaction ID ───────────────────────────────────────────────────
    for pattern in TXN_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            result["txn_id"] = matches[0]
            break

    # ── Оцениваем confidence ──────────────────────────────────────────────────
    found_count = sum([
        result["amount_found"] is not None,
        result["date_found"] is not None,
        result["time_found"] is not None,
    ])
    if found_count >= 3:
        result["ocr_confidence"] = "high"
    elif found_count >= 2:
        result["ocr_confidence"] = "medium"
    else:
        result["ocr_confidence"] = "low"

    return result


async def analyze_receipt(
    image_bytes: bytes,
    expected_amount: int,
) -> dict:
    """
    Главная функция анализа чека.

    Принимает bytes изображения и ожидаемую сумму.
    Возвращает dict с результатами OCR и вердиктом.

    Если OCR недоступен — возвращает пустой результат (без ошибки).
    """
    if not OCR_AVAILABLE:
        return {
            "ocr_available": False,
            "amount_match":  None,
            "summary":       "OCR недоступен — проверьте вручную",
        }

    try:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes))

        # Улучшаем для OCR: увеличиваем если маленькое
        w, h = img.size
        if max(w, h) < 1000:
            scale = 1000 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

        # Конвертируем в RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # OCR с поддержкой русского и английского
        text = pytesseract.image_to_string(
            img,
            lang='rus+eng',
            config='--psm 6',  # uniform block of text
        )

        result = _extract_from_text(text, expected_amount)
        result["ocr_available"] = True
        return result

    except Exception as e:
        logger.error("OCR error: %s", e)
        return {
            "ocr_available": False,
            "amount_match":  None,
            "summary":       f"OCR ошибка: {e}",
        }


def format_ocr_result(ocr: dict, expected_amount: int) -> str:
    """
    Форматирует результат OCR для отображения в сообщении администратору.
    """
    if not ocr.get("ocr_available"):
        return "🔍 OCR недоступен — проверьте вручную"

    lines = ["🔍 <b>Анализ чека (OCR)</b>"]

    # Сумма
    if ocr.get("amount_found"):
        found = ocr["amount_found"]
        match = ocr.get("amount_match")
        icon = "✅" if match else "⚠️"
        lines.append(f"{icon} Сумма: <b>{found:,.0f} сом</b> (ожидалось {expected_amount:,})")
    else:
        lines.append("❓ Сумма: не найдена")

    # Дата
    if ocr.get("date_found"):
        fresh = ocr.get("date_fresh")
        icon = "✅" if fresh else "⚠️"
        time_str = f" {ocr['time_found']}" if ocr.get("time_found") else ""
        lines.append(f"{icon} Дата: {ocr['date_found']}{time_str}")
    else:
        lines.append("❓ Дата: не найдена")

    # Имя
    if ocr.get("name_found"):
        lines.append(f"👤 Имя: {ocr['name_found']}")

    # Transaction ID
    if ocr.get("txn_id"):
        lines.append(f"🔖 ID: {ocr['txn_id']}")

    # Confidence
    conf_map = {"high": "🟢 высокий", "medium": "🟡 средний", "low": "🔴 низкий"}
    conf = conf_map.get(ocr.get("ocr_confidence", "low"), "неизвестно")
    lines.append(f"📊 Уверенность OCR: {conf}")

    return "\n".join(lines)