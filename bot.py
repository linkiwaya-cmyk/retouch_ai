"""
bot.py — Retouch Lab v3
+ бесплатная пробная обработка (1 раз)
+ inline-кнопки тарифов
+ оплата через QR
+ админ-подтверждение
+ уведомления об окончании подписки
+ PM2-ready
pipeline.py не тронут.
"""

import asyncio
import io
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BufferedInputFile,
    FSInputFile,
)
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from pipeline import process_image
# Цены в разных валютах
PLAN_PRICES_USD = {"1m": 12, "3m": 29, "6m": 59, "1y": 99}
PLAN_PRICES_VND = {"1m": 299000, "3m": 749000, "6m": 1499000, "1y": 2699000}

from database import (
    init_db,
    add_user,
    check_active_subscription,
    activate_subscription,
    update_payment_status,
    create_payment,
    get_trial_count,
    increment_trial_count,
    get_pending_payments,
    PLAN_NAMES,
    PLAN_NAMES_LANG,
    get_plan_name,
    PLAN_PRICES,
    TRIAL_LIMIT,
)
from notifications import send_expiry_notifications
from texts import t, LANGUAGES, MODES_TRANSLATED, TEXTS
from database import get_user_language, set_user_language
from broadcast import broadcast, get_all_users
from reminders import send_reminders, RELAUNCH_MESSAGE
from database import update_last_active, reset_reminder_flag
try:
    from ocr_check import analyze_receipt, format_ocr_result
    OCR_ENABLED = True
except ImportError:
    OCR_ENABLED = False
    async def analyze_receipt(*a, **k): return {"ocr_available": False}
    def format_ocr_result(*a, **k): return "🔍 OCR недоступен"

load_dotenv()

BOT_TOKEN      = os.getenv("BOT_TOKEN")

# ══════════════════════════════════════════════════════════════════════════════
# UI TEXTS — все тексты в одном месте для удобного редактирования
# ══════════════════════════════════════════════════════════════════════════════

# Welcome screen (premium/professional вариант)
WELCOME_TEXT = (
    "✦ ═══════════════════ ✦\n"
    "✨ <b>R E T O U C H   L A B</b>\n"
    "<i>AI-ретушь фото прямо в Telegram</i>\n"
    "✦ ═══════════════════ ✦\n\n"
    "Профессиональная Dodge & Burn ретушь за секунды.\n"
    "Без потери качества. Без plastic-эффекта.\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "🎨 <b>5 режимов обработки:</b>\n\n"
    "✨ Чистая кожа — минимально, натурально\n"
    "🌿 Натуральная ретушь — для каждого дня\n"
    "💫 Объём и свет — глубина, светотень\n"
    "💄 Beauty Pro — для соцсетей и Instagram\n"
    "🌟 Журнальный стиль — editorial look\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "📷 4K / 24MP / HEIC / WebP\n"
    "🎁 <b>Первые 3 фото — бесплатно</b>\n\n"
    "📌 Отправляйте фото <b>файлом</b> (📎 → Файл)"
)

# О нас / About
ABOUT_TEXT = (
    "✨ <b>RETOUCH LAB</b> — AI-ретушёр от Linkiway\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "🎯 <b>Наш подход</b>\n\n"
    "Не AI beauty filter.\n"
    "А <b>натуральная Dodge & Burn ретушь</b> —\n"
    "как делают профессиональные ретушёры вручную.\n\n"
    "✦ Текстура кожи и поры сохраняются\n"
    "✦ Родинки и особенности лица остаются\n"
    "✦ Натуральный цвет губ и тон кожи\n"
    "✦ Оригинальное разрешение 4K / 24MP\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "🎨 <b>5 режимов обработки:</b>\n\n"
    "✨ <b>Чистая кожа</b>\n"
    "Минимальная обработка. Убирает дефекты,\n"
    "сохраняет натуральность и текстуру.\n\n"
    "🌿 <b>Натуральная ретушь</b>\n"
    "Основной универсальный режим.\n"
    "Чистая кожа + лёгкое выравнивание тона.\n\n"
    "💫 <b>Объём и свет</b>\n"
    "Акцент на светотень и объём лица.\n"
    "Аккуратный Dodge & Burn.\n\n"
    "💄 <b>Beauty Pro</b>\n"
    "Качественная beauty-ретушь для соцсетей.\n"
    "Чище, выразительнее — без пластика.\n\n"
    "🌟 <b>Журнальный стиль</b>\n"
    "Самый выразительный режим.\n"
    "Editorial / magazine look.\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "⚡ 20–40 минут ручной ретуши → секунды.\n"
    "🚀 Retouch Lab постоянно развивается."
)

# Поддержка
SUPPORT_TEXT = (
    "💬 <b>Поддержка Retouch Lab</b>\n\n"
    "По вопросам:\n"
    "• подписки и оплаты\n"
    "• технических ошибок\n"
    "• сотрудничества и партнёрства\n\n"
    "Напишите нам: <b>@linkiway_support</b>\n\n"
    "Отвечаем быстро 🙌"
)

# Инструкция по отправке фото
SEND_PHOTO_TEXT = (
    "📎 <b>Отправь фото файлом для максимального качества</b>\n\n"
    "Как отправить:\n"
    "1. Нажми скрепку 📎\n"
    "2. Выбери <b>Файл</b> (не галерею)\n"
    "3. Найди фото в памяти телефона\n\n"
    "Форматы: JPG · PNG · HEIC · WebP\n"
    "Поддержка: 4K / 24MP / high-resolution"
)

# Блокировка обычного фото
PHOTO_BLOCKED_TEXT = (
    "⚠️ <b>Telegram сжимает обычные фото</b>\n\n"
    "Для сохранения полного качества отправь фото <b>файлом</b>:\n\n"
    "📎 Скрепка → <b>Файл</b> → выбери фото\n\n"
    "Это важно: Retouch Lab сохраняет оригинальное разрешение\n"
    "и возвращает файл в полном качестве — 4K, 24MP и выше."
)
ADMIN_ID       = int(os.getenv("ADMIN_CHAT_ID", "532189427"))

# ══════════════════════════════════════════════════════════════════════════════
# РЕЖИМЫ ОБРАБОТКИ — 5 пресетов
# ══════════════════════════════════════════════════════════════════════════════

MODES = {
    "clean": {
        "name": "✨ Чистая кожа",
        "desc": (
            "Минимальная обработка. Убирает прыщи и дефекты.\n"
            "Максимально натуральный результат — текстура и родинки сохраняются."
        ),
        "preset": {
            "mode": "professional",
            "tasks": [
                {"Plugin": "Heal",        "Scale": 0, "Alpha1": 0.5},
                {"Plugin": "Eye Vessels", "Scale": 0, "Alpha1": 0.6},
                {"Plugin": "Fabric",      "Scale": 0, "Alpha1": 0.12},
                {"Plugin": "Dodge Burn",  "Scale": 2, "Alpha1": 0.35, "Alpha2": 0.0},
            ]
        }
    },
    "natural": {
        "name": "🌿 Натуральная ретушь",
        "desc": (
            "Основной режим для каждого дня.\n"
            "Чистит кожу, сохраняет текстуру и поры, улучшает Dodge & Burn лица."
        ),
        "preset": {
            "mode": "professional",
            "tasks": [
                {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.28},
                {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 0.8},
                {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.35},
                {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.2,  "Alpha2": 0.2},
                {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 0.85, "Alpha2": 0.0},
            ]
        }
    },
    "depth": {
        "name": "💫 Объём и свет",
        "desc": (
            "Добавляет объём и глубину лицу.\n"
            "Усиливает Dodge & Burn — профессиональный портретный вид."
        ),
        "preset": {
            "mode": "professional",
            "tasks": [
                {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.22},
                {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 0.8},
                {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.55},
                {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.18, "Alpha2": 0.18},
                {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 1.1,  "Alpha2": 0.0},
            ]
        }
    },
    "beauty": {
        "name": "💄 Beauty Pro",
        "desc": (
            "Качественная beauty-ретушь для Instagram и соцсетей.\n"
            "Чистая кожа, выразительные глаза — без пластика и изменения цвета."
        ),
        "preset": {
            "mode": "professional",
            "tasks": [
                {"Plugin": "Heal",             "Scale": 0, "Alpha1": 0.35},
                {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.38},
                {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 0.95},
                {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.65},
                {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.28, "Alpha2": 0.28},
                {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 1.0,  "Alpha2": 0.0},
            ]
        }
    },
    "magazine": {
        "name": "🌟 Журнальный стиль",
        "desc": (
            "Премиальный editorial-режим.\n"
            "Максимально чистая кожа, выраженный объём — magazine look."
        ),
        "preset": {
            "mode": "professional",
            "tasks": [
                {"Plugin": "Heal",             "Scale": 0, "Alpha1": 0.55},
                {"Plugin": "Fabric",           "Scale": 0, "Alpha1": 0.55},
                {"Plugin": "Eye Vessels",      "Scale": 0, "Alpha1": 1.0},
                {"Plugin": "Eye Brilliance",   "Scale": 0, "Alpha1": 0.8},
                {"Plugin": "White Teeth",      "Scale": 0, "Alpha1": 0.38, "Alpha2": 0.38},
                {"Plugin": "Dodge Burn",       "Scale": 2, "Alpha1": 1.25, "Alpha2": 0.0},
            ]
        }
    },
}

DEFAULT_MODE = "natural"  # режим по умолчанию

# Текущий режим пользователя — хранится в памяти
_user_mode: dict = {}  # uid → mode_key

# Флаг активной акции — устанавливается через /promo
# promo_active_until = None или datetime когда акция заканчивается
_promo_until: float = 0.0  # unix timestamp конца акции
# GROUP_CHAT_ID — отдельная группа куда приходят чеки с кнопками approve/reject
# Получить: создай группу → добавь бота → напиши /start → смотри getUpdates
GROUP_CHAT_ID  = int(os.getenv("GROUP_CHAT_ID", os.getenv("ADMIN_CHAT_ID", "532189427")))
QR_PATH         = Path(__file__).parent / "qr_code.png"       # Бакай Банк RU/KY
QR_PATH_VI      = Path(__file__).parent / "qr_code_vi.png"    # Vietcombank VI
QR_PATH_USDT    = Path(__file__).parent / "qr_code_usdt.png"  # USDT
USDT_QR_PATH    = QR_PATH_USDT  # алиас для callback_show_usdt_qr
USDT_ADDRESS    = "TVjWpiVhRBDQKKFBn8KzP4Mc7noRYoLFFZ"
MBANK_PHONE     = "+996 (500) 070 759"
MBANK_NAME      = "АЛИНА А."
VIETCOMBANK_NUM = "QRGD00010667377180"
VIETCOMBANK_NAME = "ABDURASULOVA ALINA"


def get_qr_path(lang: str) -> Path:
    """Возвращает нужный QR в зависимости от языка."""
    if lang == "vi" and QR_PATH_VI.exists():
        return QR_PATH_VI
    if lang == "en" and QR_PATH_USDT.exists():
        return QR_PATH_USDT
    return QR_PATH


def get_payment_caption(lang: str, plan_name: str, amount_som: int,
                         amount_usd: int, amount_vnd: int) -> str:
    """Текст инструкции по оплате на нужном языке."""
    if lang == "kk":
        kzt_map = {11: 5800, 28: 14700, 57: 29900, 102: 53500}
        amount_kzt = kzt_map.get(amount_usd, amount_usd * 525)
        return (
            f"💳 <b>Төлем: {plan_name} — {amount_kzt:,} теңге (~${amount_usd})</b>\n\n"
            f"<b>{amount_kzt:,} теңге</b> Бакай Банкке аударыңыз:\n\n"
            f"👤 <b>{MBANK_NAME}</b>\n"
            f"📱 <b>{MBANK_PHONE}</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📌 <b>Төлем жасау:</b>\n"
            "1. Бакай Банк қолданбасын ашыңыз\n"
            "2. Аударымдар → Телефон нөмірі бойынша\n"
            f"3. Сумма: <b>{amount_kzt:,} теңге (~${amount_usd})</b>\n"
            "4. Аударыңыз және чекті сақтаңыз\n"
            "5. Скриншотты осы жерге жіберіңіз\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "⏳ Жазылым бірнеше минут ішінде белсендіріледі."
        )
    elif lang == "vi":
        return (
            f"💳 <b>Thanh toán: {plan_name} — {amount_vnd:,} VND (~${amount_usd})</b>\n\n"
            f"Chuyển <b>{amount_vnd:,} VND</b> qua Vietcombank:\n\n"
            f"👤 <b>{VIETCOMBANK_NAME}</b>\n"
            f"🏦 Số TK: <b>{VIETCOMBANK_NUM}</b>\n"
            f"🏢 CN Nha Trang\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📌 <b>Hướng dẫn:</b>\n"
            "1. Mở ứng dụng ngân hàng\n"
            "2. Quét mã QR hoặc chuyển khoản thủ công\n"
            f"3. Số tiền: <b>{amount_vnd:,} VND</b>\n"
            "4. Ghi chú: <b>retouch</b>\n"
            "5. Gửi ảnh chụp màn hình tại đây 👇\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "⏳ Đăng ký sẽ được kích hoạt trong vài phút."
        )
    elif lang == "en":
        return (
            f"💳 <b>Payment: {plan_name} — ${amount_usd}</b>\n\n"
            f"Send <b>${amount_usd} USDT</b> (TRC20):\n\n"
            f"💎 <b>Wallet address:</b>\n"
            f"<code>{USDT_ADDRESS}</code>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📌 <b>How to pay:</b>\n"
            "1. Open your crypto wallet\n"
            "2. Send USDT via TRC20 network\n"
            f"3. Amount: <b>${amount_usd} USDT</b>\n"
            "4. Send screenshot here 👇\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "⚠️ <b>Important:</b> use TRC20 network only!\n"
            "⏳ Subscription activates within a few minutes."
        )
    else:
        # RU — рубли
        return (
            f"💳 <b>Оплата: {plan_name} — {amount_som:,} руб</b>\n\n"
            f"Переведите <b>{amount_som:,} руб</b> в Бакай Банк:\n\n"
            f"👤 <b>{MBANK_NAME}</b>\n"
            f"📱 <b>{MBANK_PHONE}</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📌 <b>Как оплатить:</b>\n"
            "1. Откройте Бакай Банк\n"
            "2. Переводы → По номеру телефона\n"
            f"3. Сумма: <b>{amount_som:,} сом (~${amount_usd})</b>\n"
            "4. Переведите и сохраните чек\n"
            "5. Отправьте скриншот сюда 👇\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "⏳ Подписка активируется в течение нескольких минут."
        )

# Очередь обработки — каждый пользователь имеет свою очередь
# Фото обрабатываются по одному, остальные ждут автоматически
_queues: dict = {}       # uid → asyncio.Queue
_queue_active: set = set()  # uid → обрабатывается прямо сейчас


def _get_queue(uid: int) -> "asyncio.Queue":
    if uid not in _queues:
        _queues[uid] = asyncio.Queue()
    return _queues[uid]


async def _queue_worker(uid: int, bot_instance):
    """Воркер очереди — обрабатывает фото по одному для каждого пользователя."""
    q = _get_queue(uid)
    while not q.empty():
        data, filename, message = await q.get()
        try:
            # Проверяем доступ перед КАЖДЫМ фото из очереди
            if await _can_process(message):
                await _do_process(message, data, filename)
        except Exception as e:
            logger.error("Queue worker error uid=%d: %s", uid, e)
        finally:
            q.task_done()
    _queue_active.discard(uid)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())


# ── FSM ───────────────────────────────────────────────────────────────────────
class LanguageStates(StatesGroup):
    choosing = State()


class PaymentStates(StatesGroup):
    waiting_screenshot = State()


class BroadcastStates(StatesGroup):
    waiting_text    = State()   # ждём текст рассылки
    waiting_confirm = State()   # ждём подтверждение


# ── Клавиатуры ────────────────────────────────────────────────────────────────
main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="✨ Обработать фото")],
        [KeyboardButton(text="🎁 Попробовать бесплатно"), KeyboardButton(text="💎 Подписка")],
        [KeyboardButton(text="🎥 Примеры до / после"), KeyboardButton(text="ℹ️ О боте")],
        [KeyboardButton(text="💬 Поддержка"), KeyboardButton(text="🌐 Язык / Language")],
    ],
    resize_keyboard=True,
)

modes_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="✨ Чистая кожа")],
        [KeyboardButton(text="🌿 Натуральная ретушь")],
        [KeyboardButton(text="💫 Объём и свет")],
        [KeyboardButton(text="💄 Beauty Pro")],
        [KeyboardButton(text="🌟 Журнальный стиль")],
        [KeyboardButton(text="📖 О режимах"), KeyboardButton(text="⬅️ Главное меню")],
    ],
    resize_keyboard=True,
)

# Клавиатура после выбора режима — "Назад к режимам"
back_to_modes = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="⬅️ Назад к режимам")]],
    resize_keyboard=True,
)


def make_modes_keyboard(lang: str = "ru") -> ReplyKeyboardMarkup:
    """Клавиатура режимов на нужном языке."""
    from texts import TEXTS
    _l = lang if lang in ("ru", "en", "vi", "ky", "kk") else "ru"
    def _t(key):
        return TEXTS.get(key, {}).get(_l) or TEXTS.get(key, {}).get("ru", "")
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=_t("btn_mode_clean"))],
            [KeyboardButton(text=_t("btn_mode_natural"))],
            [KeyboardButton(text=_t("btn_mode_depth"))],
            [KeyboardButton(text=_t("btn_mode_beauty"))],
            [KeyboardButton(text=_t("btn_mode_magazine"))],
            [KeyboardButton(text=_t("btn_about_modes")),
             KeyboardButton(text=_t("btn_back_main"))],
        ],
        resize_keyboard=True,
    )


def make_back_to_modes(lang: str = "ru") -> ReplyKeyboardMarkup:
    txt = BACK_TO_MODES_TEXTS.get(lang, BACK_TO_MODES_TEXTS["ru"])
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=txt)]],
        resize_keyboard=True,
    )


def make_main_menu(lang: str = "ru", remaining: int = 0,
                   has_sub: bool = False, promo_active: bool = False) -> ReplyKeyboardMarkup:
    """Главное меню на нужном языке."""
    from texts import TEXTS
    rows = []

    # Trial кнопка — только если есть попытки
    if not has_sub and remaining > 0:
        process_text = TEXTS["btn_process"][lang].replace("✨ ", "")
        rows.append([KeyboardButton(text=f"🎁 {process_text} ({remaining}/{TRIAL_LIMIT})")])

    # Кнопка акции
    if promo_active and not has_sub:
        rows.append([KeyboardButton(text="🔥 Акция — 799 сом (~$9)")])

    # Основные кнопки
    rows += [
        [KeyboardButton(text=TEXTS["btn_process"].get(lang, TEXTS["btn_process"]["ru"]))],
        [KeyboardButton(text=TEXTS["btn_examples"].get(lang, TEXTS["btn_examples"]["ru"])),
         KeyboardButton(text=TEXTS["btn_subscription"].get(lang, TEXTS["btn_subscription"]["ru"]))],
        [KeyboardButton(text=TEXTS["btn_support"].get(lang, TEXTS["btn_support"]["ru"])),
         KeyboardButton(text=TEXTS.get("btn_language", {}).get(lang) or TEXTS.get("btn_language", {}).get("en", "🌐 Language"))],
        [KeyboardButton(text=TEXTS["btn_about"].get(lang, TEXTS["btn_about"]["ru"]))],
    ]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)

back_menu = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="⬅️ Назад")]],
    resize_keyboard=True,
)

BACK_TEXTS = {
    "ru": "⬅️ Назад",
    "en": "⬅️ Back",
    "vi": "⬅️ Quay lại",
    "ky": "⬅️ Артка",
    "kk": "⬅️ Артқа",
}

BACK_TO_MODES_TEXTS = {
    "ru": "⬅️ Назад к режимам",
    "en": "⬅️ Back to modes",
    "vi": "⬅️ Quay lại chế độ",
    "ky": "⬅️ Режимдерге кайт",
    "kk": "⬅️ Режимдерге қайту",
}

def make_back_menu(lang: str = "ru") -> ReplyKeyboardMarkup:
    """Кнопка Назад на языке пользователя."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BACK_TEXTS.get(lang, "⬅️ Назад"))]],
        resize_keyboard=True,
    )


def plans_keyboard(lang: str = "ru") -> InlineKeyboardMarkup:
    """Кнопки тарифов на языке пользователя."""
    if lang == "en":
        labels = [
            ("📅 1 month — $12",              "buy_1m"),
            ("📅 3 months — $29 · -15%",       "buy_3m"),
            ("📅 6 months — $59 · -25%",       "buy_6m"),
            ("📅 1 year — $99 · -35% 🔥",     "buy_1y"),
        ]
    elif lang == "vi":
        labels = [
            ("📅 1 tháng — 299,000 VND (~$12)",          "buy_1m"),
            ("📅 3 tháng — 749,000 VND (~$29) · -15%",   "buy_3m"),
            ("📅 6 tháng — 1,499,000 VND (~$59) · -25%", "buy_6m"),
            ("📅 1 năm — 2,699,000 VND (~$99) · -35% 🔥","buy_1y"),
        ]
    elif lang == "ky":
        labels = [
            ("📅 1 ай — 999 сом (~$12)",              "buy_1m"),
            ("📅 3 ай — 2,490 сом (~$29) · -15%",     "buy_3m"),
            ("📅 6 ай — 4,999 сом (~$59) · -25%",     "buy_6m"),
            ("📅 1 жыл — 8,999 сом (~$99) · -35% 🔥","buy_1y"),
        ]
    elif lang == "kk":
        labels = [
            ("📅 1 ай — 5,999 теңге (~$12)",              "buy_1m"),
            ("📅 3 ай — 14,999 теңге (~$29) · -15%",     "buy_3m"),
            ("📅 6 ай — 29,999 теңге (~$59) · -25%",     "buy_6m"),
            ("📅 1 жыл — 53,999 теңге (~$99) · -35% 🔥","buy_1y"),
        ]
    else:  # ru
        labels = [
            (f"📅 1 месяц — {PLAN_PRICES['1m']:,} руб",       "buy_1m"),
            (f"📅 3 месяца — {PLAN_PRICES['3m']:,} руб · -15%","buy_3m"),
            (f"📅 6 месяцев — {PLAN_PRICES['6m']:,} руб · -25%","buy_6m"),
            (f"📅 1 год — {PLAN_PRICES['1y']:,} руб · -35% 🔥","buy_1y"),
        ]
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=text, callback_data=cb)]
        for text, cb in labels
    ])


def buy_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💎 Купить подписку", callback_data="open_plans")]
    ])


# ══════════════════════════════════════════════════════════════════════════════
# /start
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await add_user(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
        first_name=message.from_user.first_name,
    )
    # Определяем язык — если уже выбран вручную, не меняем
    lang = await get_user_language(message.from_user.id)
    if not lang or lang == "ru":
        # Для новых пользователей — определяем по Telegram language_code
        tg_lang = (message.from_user.language_code or "").lower()
        if tg_lang.startswith("ky"):
            auto_lang = "ky"
        elif tg_lang.startswith("ru"):
            auto_lang = "ru"
        elif tg_lang.startswith("kk"):
            auto_lang = "kk"
        elif tg_lang.startswith("vi"):
            auto_lang = "vi"
        elif tg_lang.startswith("en"):
            auto_lang = "en"
        else:
            auto_lang = "en"  # fallback для всех остальных
        # Сохраняем только если ещё не установлен вручную
        if not lang:
            await set_user_language(message.from_user.id, auto_lang)
            lang = auto_lang
        # Если lang == "ru" и это старый пользователь — оставляем ru
    # Динамическое приветствие
    uid = message.from_user.id
    has_sub = bool(await check_active_subscription(uid))
    trial_used = await get_trial_count(uid)
    remaining = max(0, TRIAL_LIMIT - trial_used)

    import time as _time
    promo_active = _promo_until > _time.time()

    if has_sub:
        trial_btn = "✨ Обработать фото"
    elif remaining > 0:
        # Если есть бесплатные попытки — показываем их (акция не скрывает)
        trial_btn = f"🎁 Попробовать бесплатно (осталось {remaining} из {TRIAL_LIMIT})"
    elif promo_active:
        # Бесплатные закончились + акция активна
        trial_btn = "🔥 Акция — 799 сом (~$9)"
    else:
        trial_btn = "💎 Купить подписку"

    # Строим меню — кнопка обработки всегда есть
    # Если есть trial или подписка — первая кнопка ведёт к режимам
    # Если акция — отдельная кнопка акции добавляется
    # Строим меню
    menu_rows = []

    # Кнопка trial — только если есть бесплатные попытки
    if not has_sub and remaining > 0:
        menu_rows.append([KeyboardButton(text=f"🎁 Обработать фото (осталось {remaining} из {TRIAL_LIMIT})")])

    # Кнопка акции — только если активна и нет подписки
    if promo_active and not has_sub:
        menu_rows.append([KeyboardButton(text="🔥 Акция — 799 сом (~$9)")])

    # Основные кнопки — всегда
    menu_rows += [
        [KeyboardButton(text="✨ Обработать фото")],
        [KeyboardButton(text="🎥 Примеры до / после"), KeyboardButton(text="💎 Подписка")],
        [KeyboardButton(text="💬 Поддержка"), KeyboardButton(text="🌐 Язык / Language")],
        [KeyboardButton(text="ℹ️ О боте")],
    ]

    lang = await get_user_language(message.from_user.id)
    dynamic_menu = make_main_menu(lang, remaining, has_sub, promo_active)

    # Баннер если есть
    banner_path = Path(__file__).parent / "start_banner.png"
    if banner_path.exists():
        try:
            await message.answer_photo(
                photo=FSInputFile(str(banner_path)),
                caption="✨ <b>RETOUCH LAB</b>",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.warning("Banner send failed: %s", e)

    await message.answer(
        WELCOME_TEXT,
        reply_markup=dynamic_menu,
        parse_mode="HTML",
    )

    # Промо-блок для пользователей без подписки которые уже пробовали бота
    if not has_sub and trial_used > 0 and remaining == 0:
        promo_kb = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="💎 Выбрать подписку", callback_data="open_plans")
        ]])
        from texts import TEXTS as _TPB
        _pblang = await get_user_language(uid)
        await message.answer(
            _TPB["promo_block"].get(_pblang, _TPB["promo_block"]["ru"]),
            reply_markup=promo_kb,
            parse_mode="HTML",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Меню
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(F.text.in_({"🔥 Акция — 799 сом (~$9)", "🔥 999 сомға сатып алу (~$9)"}))
async def menu_promo_start(message: Message, state: FSMContext):
    """Кнопка акции в главном меню — показывает акционное предложение."""
    import time as _time
    if _promo_until <= _time.time():
        _plang = await get_user_language(message.from_user.id)
        from texts import TEXTS as _TP
        await message.answer(_TP["promo_ended"].get(_plang, "⏰ Акция уже завершена."), reply_markup=main_menu)
        return

    promo_buy_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="🔥 Купить за 799 сом (~$9)",
            callback_data="buy_promo_1m"
        )
    ]])

    import datetime
    ends_at = datetime.datetime.fromtimestamp(_promo_until).strftime("%H:%M")

    _pslang = await get_user_language(message.from_user.id)
    from texts import TEXTS as _TPS
    _ps_btn = _TPS["promo_buy_btn"].get(_pslang, _TPS["promo_buy_btn"]["ru"])
    # Кнопка на языке пользователя
    promo_btn_labels = {
        "ru": "🔥 Купить за 799 руб",
        "ky": "🔥 799 сомго сатып ал",
        "kk": "🔥 4,500 теңгеге сатып алу",
        "en": "🔥 Buy for $9 USDT",
        "vi": "🔥 Mua với giá 250,000 VND",
    }
    _ps_btn = promo_btn_labels.get(_pslang, "🔥 Купить")
    promo_buy_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=_ps_btn, callback_data="buy_promo_1m")
    ]])
    await message.answer(
        _TPS["promo_screen"].get(_pslang, _TPS["promo_screen"]["ru"]).format(ends_at=ends_at),
        reply_markup=promo_buy_kb,
        parse_mode="HTML",
    )


@dp.message(F.text.in_({"🌐 Язык / Language", "🌐 Язык", "🌐 Language", "🌐 Ngôn ngữ", "🌐 Тилди өзгөртүү", "🌐 Тіл"}))
async def menu_language(message: Message, state: FSMContext):
    """Кнопка смены языка в главном меню."""
    lang = await get_user_language(message.from_user.id)
    from texts import TEXTS
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🇰🇬 Кыргызча",   callback_data="lang_ky")],
        [InlineKeyboardButton(text="🇷🇺 Русский",    callback_data="lang_ru")],
        [InlineKeyboardButton(text="🇰🇿 Қазақша",    callback_data="lang_kk")],
        [InlineKeyboardButton(text="🇬🇧 English",    callback_data="lang_en")],
        [InlineKeyboardButton(text="🇻🇳 Tiếng Việt", callback_data="lang_vi")],
    ])
    await message.answer(
        TEXTS["choose_lang"].get(lang, TEXTS["choose_lang"]["ru"]),
        reply_markup=kb,
        parse_mode="HTML",
    )


@dp.message(F.text.in_({'⬅️ Назад', '⬅️ Back', '⬅️ Quay lại', '⬅️ Артка', '⬅️ Артқа', '⬅️ Башкы меню', '⬅️ Басты мәзір', '⬅️ Menu chính', '⬅️ Главное меню', '⬅️ Main menu'}))
async def back(message: Message, state: FSMContext):
    await state.clear()
    uid = message.from_user.id
    user_lang = await get_user_language(uid)
    has_sub = bool(await check_active_subscription(uid))
    trial_used = await get_trial_count(uid)
    remaining = max(0, TRIAL_LIMIT - trial_used)
    import time as _time
    promo_active = _promo_until > _time.time()
    from texts import TEXTS as _TB
    await message.answer(
        _TB["main_menu_title"].get(user_lang, "Главное меню"),
        reply_markup=make_main_menu(user_lang, remaining, has_sub, promo_active)
    )


@dp.message(F.text.in_({'⬅️ Back to modes', '⬅️ Quay lại chế độ', '⬅️ Режимдерге кайт', '⬅️ Режимдерге қайту', '⬅️ Назад к режимам'}))
async def back_to_modes_handler(message: Message, state: FSMContext):
    """Возврат к списку режимов — НЕ в главное меню."""
    user_lang = await get_user_language(message.from_user.id)
    from texts import TEXTS as _TBM
    await message.answer(
        _TBM["choose_mode"].get(user_lang, _TBM["choose_mode"]["ru"]),
        reply_markup=make_modes_keyboard(user_lang),
    )


@dp.message(F.text.in_({'ℹ️ About', 'ℹ️ Giới thiệu', 'ℹ️ Бот жөнүндө', 'ℹ️ Бот туралы', 'ℹ️ О боте'}))
async def menu_about(message: Message):
    from texts import TEXTS
    lang = await get_user_language(message.from_user.id)
    txt = TEXTS["about_full"].get(lang, TEXTS["about_full"]["ru"])
    await message.answer(txt, reply_markup=make_back_menu(lang), parse_mode="HTML")


@dp.message(F.text.in_({'💬 Колдоо', '💬 Қолдау', '💬 Hỗ trợ', '💬 Поддержка', '💬 Support'}))
async def menu_support(message: Message):
    from texts import TEXTS
    lang = await get_user_language(message.from_user.id)
    txt = TEXTS["support_text"].get(lang, TEXTS["support_text"]["ru"])
    await message.answer(txt, reply_markup=back_menu, parse_mode="HTML")


@dp.message(F.text.in_({'🎥 Before / After examples', '🎥 Примеры до / после', '🎥 Мисалдар чейин / кийин', '🎥 Мысалдар дейін / кейін', '🖼 Before / After', '🎥 Ví dụ trước / sau'}))
async def menu_before_after(message: Message):
    from aiogram.types import InputMediaPhoto

    # Проверяем все возможные пути к видео
    video_paths = [
        Path(__file__).parent / "examples_video" / "before_after.mp4",
        Path(__file__).parent / "examples" / "before_after.mp4",
    ]
    video_path = next((p for p in video_paths if p.exists()), None)

    # Папки с фото
    examples_dirs = [
        Path(__file__).parent / "examples_video",
        Path(__file__).parent / "examples",
    ]

    from texts import TEXTS as _TBA
    _balang = await get_user_language(message.from_user.id)
    DESCRIPTION = _TBA["before_after_desc"].get(_balang, _TBA["before_after_desc"]["ru"])

    # ── Приоритет 1: видео ────────────────────────────────────────────────────
    if video_path:
        logger.info("[VIDEO] sending: %s (%.1f MB)",
                    video_path, video_path.stat().st_size / 1024 / 1024)
        try:
            await message.answer_video(
                video=FSInputFile(str(video_path)),
                caption=DESCRIPTION,
                parse_mode="HTML",
                reply_markup=back_menu,
                supports_streaming=True,
                width=1080,
                height=1920,
            )
            logger.info("[VIDEO] sent OK")
        except Exception as e:
            logger.warning("[VIDEO] answer_video failed: %s — trying document", e)
            try:
                await message.answer_document(
                    document=FSInputFile(str(video_path)),
                    caption=DESCRIPTION,
                    parse_mode="HTML",
                    reply_markup=back_menu,
                )
                logger.info("[VIDEO] sent as document OK")
            except Exception as e2:
                logger.error("[VIDEO] document also failed: %s", e2)
                await message.answer(
                    DESCRIPTION,
                    reply_markup=back_menu,
                    parse_mode="HTML",
                )
        return

    # ── Приоритет 2: фото ────────────────────────────────────────────────────
    photo_files = []
    for d in examples_dirs:
        if d.exists():
            photo_files = (
                sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            )[:4]
            if photo_files:
                break

    if photo_files:
        try:
            media = [
                InputMediaPhoto(
                    media=FSInputFile(str(p)),
                    caption=DESCRIPTION if i == 0 else None,
                    parse_mode="HTML" if i == 0 else None,
                )
                for i, p in enumerate(photo_files)
            ]
            await message.answer_media_group(media=media)
            pass  # убрали лишний стикер
        except Exception as e:
            logger.error("[BEFORE_AFTER] photo send failed: %s", e)
            await message.answer(DESCRIPTION, reply_markup=back_menu, parse_mode="HTML")
        return

    # ── Fallback: нет ни видео ни фото ───────────────────────────────────────
    from texts import TEXTS
    _lang = await get_user_language(message.from_user.id)
    await message.answer(
        TEXTS["examples_updating"].get(_lang, TEXTS["examples_updating"]["ru"]),
        reply_markup=back_menu,
        parse_mode="HTML",
    )


@dp.message(F.text == "📂 Форматы")
async def menu_formats(message: Message):
    from texts import TEXTS as _TFM
    _fmlang = await get_user_language(message.from_user.id)
    await message.answer(
        _TFM["formats_text"].get(_fmlang, _TFM["formats_text"]["ru"]),
        reply_markup=make_back_menu(_fmlang),
        parse_mode="HTML",
    )


@dp.message(F.text.in_({'📖 About modes', '📖 О режимах', '📖 Режимдер жөнүндө', '📖 Режимдер туралы', '📖 Về các chế độ'}))
async def menu_about_modes(message: Message):
    user_lang = await get_user_language(message.from_user.id)
    from texts import TEXTS as _TAM
    await message.answer(
        _TAM["about_modes_text"].get(user_lang, _TAM["about_modes_text"]["ru"]),

        reply_markup=make_modes_keyboard(user_lang),
        parse_mode="HTML",
    )


@dp.message(F.text.in_({
    # RU
    "✨ Чистая кожа", "🌿 Натуральная ретушь",
    "💫 Объём и свет", "💄 Beauty Pro", "🌟 Журнальный стиль",
    # EN
    "✨ Clean Skin", "🌿 Natural Retouch",
    "💫 Depth & Light", "🌟 Magazine Style",
    # VI
    "✨ Da Sạch", "🌿 Retouch Tự Nhiên",
    "💫 Chiều Sâu & Ánh Sáng", "🌟 Phong Cách Tạp Chí",
    # KY
    "✨ Таза тери", "🌿 Табигый ретушь",
    "💫 Көлөм жана жарык", "🌟 Журнал стили",
    # KK
    "✨ Таза тері", "🌿 Табиғи ретушь",
    "💫 Көлем мен жарық", "💫 Көлем және жарық",
    "🌟 Журнал стилі",
}))
async def select_mode(message: Message):
    mode_map = {
        # RU
        "✨ Чистая кожа": "clean", "🌿 Натуральная ретушь": "natural",
        "💫 Объём и свет": "depth", "💄 Beauty Pro": "beauty",
        "🌟 Журнальный стиль": "magazine",
        # EN
        "✨ Clean Skin": "clean", "🌿 Natural Retouch": "natural",
        "💫 Depth & Light": "depth", "💄 Beauty Pro": "beauty",
        "🌟 Magazine Style": "magazine",
        # VI
        "✨ Da Sạch": "clean", "🌿 Retouch Tự Nhiên": "natural",
        "💫 Chiều Sâu & Ánh Sáng": "depth", "💄 Beauty Pro": "beauty",
        "🌟 Phong Cách Tạp Chí": "magazine",
        # KY
        "✨ Таза тери": "clean", "🌿 Табигый ретушь": "natural",
        "💫 Көлөм жана жарык": "depth", "💄 Beauty Pro": "beauty",
        "🌟 Журнал стили": "magazine",
        # KK
        "✨ Таза тері": "clean", "🌿 Табиғи ретушь": "natural",
        "💫 Көлем мен жарық": "depth", "💫 Көлем және жарық": "depth",
        "💄 Beauty Pro": "beauty",
        "🌟 Журнал стилі": "magazine",
    }
    uid = message.from_user.id
    mode_key = mode_map.get(message.text, DEFAULT_MODE)
    _user_mode[uid] = mode_key
    mode = MODES[mode_key]

    user_lang = await get_user_language(uid)
    from texts import TEXTS as _TM, MODES_TRANSLATED as _MT
    _mode_tr = _MT.get(mode_key, {}).get(user_lang, _MT.get(mode_key, {}).get("ru", {}))
    _mname = _mode_tr.get("name", mode["name"])
    _mdesc = _mode_tr.get("desc", mode["desc"])
    mode_txt = _TM["mode_selected"].get(user_lang, _TM["mode_selected"]["ru"])
    await message.answer(
        mode_txt.format(name=_mname, desc=_mdesc),
        reply_markup=make_back_to_modes(user_lang),
        parse_mode="HTML",
    )


@dp.message(F.text.in_({"🎁 Попробовать бесплатно", "🎁 Тегін байқап көру"}))
async def menu_try_free(message: Message):
    uid = message.from_user.id
    has_sub = bool(await check_active_subscription(uid))
    trial_used = await get_trial_count(uid)
    remaining = max(0, TRIAL_LIMIT - trial_used)

    if has_sub:
        _tflang = await get_user_language(uid)
        from texts import TEXTS as _TTF
        await message.answer(
            _TTF["try_free_has_sub"].get(_tflang, _TTF["try_free_has_sub"]["ru"]),
            reply_markup=make_modes_keyboard(_tflang),
            parse_mode="HTML",
        )
    elif remaining > 0:
        _trlang = await get_user_language(uid)
        from texts import TEXTS as _TTR
        await message.answer(
            _TTR["try_free_remaining"].get(_trlang, _TTR["try_free_remaining"]["ru"]).format(n=remaining, total=TRIAL_LIMIT),
            reply_markup=make_modes_keyboard(_trlang),
            parse_mode="HTML",
        )
    else:
        _telang = await get_user_language(uid)
        from texts import TEXTS as _TTE
        await message.answer(
            _TTE["paywall_full"].get(_telang, _TTE["paywall_full"]["ru"]),
            reply_markup=plans_keyboard(_telang),
            parse_mode="HTML",
        )


# Динамическая кнопка "Попробовать бесплатно (осталось X из 3)"
@dp.message(F.text.startswith("🎁"))
async def menu_try_free_dynamic(message: Message):
    """Кнопка с остатком бесплатных — ведёт к режимам."""
    uid = message.from_user.id
    has_sub = bool(await check_active_subscription(uid))
    trial_used = await get_trial_count(uid)
    remaining = max(0, TRIAL_LIMIT - trial_used)

    if has_sub or remaining > 0:
        current_mode = _user_mode.get(uid, DEFAULT_MODE)
        mode = MODES[current_mode]
        user_lang = await get_user_language(uid)
        from texts import TEXTS as _TFR, MODES_TRANSLATED as _MFR
        _mc_tr2 = _MFR.get(current_mode, {}).get(user_lang, _MFR.get(current_mode, {}).get("ru", {}))
        _mname2 = _mc_tr2.get("name", mode["name"])
        _tfr_tpl = _TFR["trial_free_remaining"].get(user_lang, _TFR["trial_free_remaining"]["ru"])
        await message.answer(
            _tfr_tpl.format(n=remaining, total=TRIAL_LIMIT, name=_mname2),
            reply_markup=make_modes_keyboard(user_lang),
            parse_mode="HTML",
        )
    else:
        _dellang = await get_user_language(uid)
        from texts import TEXTS as _TDEL
        await message.answer(
            _TDEL["paywall_full"].get(_dellang, _TDEL["paywall_full"]["ru"]),
            reply_markup=plans_keyboard(_dellang),
            parse_mode="HTML",
        )


@dp.message(F.text.in_({'📸 Обработать фото', '✨ Сүрөттү иштет', '✨ Суретті өңдеу', '✨ Обработать фото', '✨ Xử lý ảnh', '✨ Process photo'}))
async def menu_process(message: Message):
    uid = message.from_user.id
    has_sub = bool(await check_active_subscription(uid))
    trial_used = await get_trial_count(uid)
    remaining = max(0, TRIAL_LIMIT - trial_used)

    # Если нет подписки и лимит исчерпан — показываем блокировку
    if not has_sub and remaining == 0:
        _mplang = await get_user_language(uid)
        from texts import TEXTS as _TMP
        await message.answer(
            _TMP["paywall_full"].get(_mplang, _TMP["paywall_full"]["ru"]),
            reply_markup=plans_keyboard(_mplang),
            parse_mode="HTML",
        )
        return

    # Иначе — показываем режимы
    user_lang = await get_user_language(uid)
    current_mode = _user_mode.get(uid, DEFAULT_MODE)
    mode = MODES[current_mode]
    from texts import TEXTS as _TC, MODES_TRANSLATED as _MCT
    _mc_tr = _MCT.get(current_mode, {}).get(user_lang, _MCT.get(current_mode, {}).get("ru", {}))
    _mcname = _mc_tr.get("name", mode["name"])
    _mcdesc = _mc_tr.get("desc", mode["desc"])
    _cur_tpl = _TC["current_mode"].get(user_lang, _TC["current_mode"]["ru"])
    await message.answer(
        _cur_tpl.format(name=_mcname, desc=_mcdesc),
        reply_markup=make_modes_keyboard(user_lang),
        parse_mode="HTML",
    )


@dp.message(F.text.in_({'💎 Đăng ký', '💎 Subscription', '💎 Жазылуу', '💎 Жазылым', '💎 Подписка'}))
async def menu_sub(message: Message):
    sub = await check_active_subscription(message.from_user.id)

    if sub:
        end_pretty = ".".join(reversed(sub["end_date"][:10].split("-")))
        from texts import TEXTS as _TSA
        _salang = await get_user_language(message.from_user.id)
        _sa_tpl = _TSA["sub_active_msg"].get(_salang, _TSA["sub_active_msg"]["ru"])
        await message.answer(
            _sa_tpl.format(
                plan=get_plan_name(sub["plan_type"], _salang),
                date=end_pretty
            ),
            reply_markup=back_menu,
            parse_mode="HTML",
        )
        return

    from texts import TEXTS as _TSP
    lang = await get_user_language(message.from_user.id)
    plans_text = _TSP["sub_plans"].get(lang, _TSP["sub_plans"].get("ru"))
    await message.answer(plans_text, reply_markup=plans_keyboard(lang), parse_mode="HTML")


# ══════════════════════════════════════════════════════════════════════════════
# Выбор тарифа → реквизиты
# ══════════════════════════════════════════════════════════════════════════════

@dp.callback_query(F.data == "open_plans")
async def callback_open_plans(callback: CallbackQuery):
    await callback.answer()
    lang = await get_user_language(callback.from_user.id)
    from texts import TEXTS as _TOP
    plans_text = _TOP["sub_plans"].get(lang, _TOP["sub_plans"]["ru"])
    await callback.message.answer(
        plans_text,
        reply_markup=plans_keyboard(lang),
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "buy_promo_1m")
async def callback_buy_promo(callback: CallbackQuery, state: FSMContext):
    """Акционная покупка — 799 сом. Работает только через /promo рассылку."""
    uid = callback.from_user.id

    # Проверяем — если уже есть подписка, кнопка не работает
    if await check_active_subscription(uid):
        await callback.answer("У вас уже есть активная подписка! ✅", show_alert=True)
        return

    PROMO_PRICE = 799
    plan_type = "1m"
    plan_name = "1 месяц (акция 🔥)"

    await state.update_data(plan_type=plan_type)
    await state.set_state(PaymentStates.waiting_screenshot)
    await callback.answer()

    caption = (
        f"🔥 <b>Акция: {plan_name} — {PROMO_PRICE} сом (~$9)</b>\n\n"
        f"Переведите <b>{PROMO_PRICE} сом</b> в Бакай Банк:\n\n"
        f"👤 <b>{MBANK_NAME}</b>\n"
        f"📱 <b>{MBANK_PHONE}</b>\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📌 <b>Как оплатить:</b>\n"
        "1. Откройте Бакай Банк\n"
        "2. Переводы → По номеру телефона\n"
        f"3. Введите сумму: <b>{PROMO_PRICE} сом (~$9)</b>\n"
        "4. Переведите и сохраните чек\n"
        "5. Отправьте скриншот сюда 👇\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "⏳ Подписка активируется в течение нескольких минут."
    )

    lang = await get_user_language(callback.from_user.id)
    qr = get_qr_path(lang)
    from texts import TEXTS as _TCK

    # 1. Банковский QR
    if qr.exists():
        try:
            await callback.message.answer_photo(
                photo=FSInputFile(qr),
                caption=caption,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("QR send error: %s", e)
            await callback.message.answer(caption, parse_mode="HTML")
    else:
        await callback.message.answer(caption, parse_mode="HTML")
        qr_upd = _TCK["qr_updating"].get(lang, _TCK["qr_updating"]["ru"])
        await callback.message.answer(qr_upd)

    # 2. USDT блок
    usdt_txt = _TCK["usdt_payment"].get(lang, _TCK["usdt_payment"]["en"])
    usdt_btn_txt = _TCK["usdt_btn"].get(lang, "📱 USDT QR")
    usdt_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=usdt_btn_txt, callback_data="show_usdt_qr")
    ]])
    await callback.message.answer(
        usdt_txt.format(address=USDT_ADDRESS),
        reply_markup=usdt_kb,
        parse_mode="HTML",
    )

    # 3. Visa/MC текст
    _card_txt = _TCK.get("card_coming_soon", {})
    if isinstance(_card_txt, dict):
        _card_txt = _card_txt.get(lang, _card_txt.get("ru", ""))
    if _card_txt:
        await callback.message.answer(_card_txt, parse_mode="HTML")

    await callback.message.answer(_TCK["send_check"].get(lang, _TCK["send_check"]["ru"]))


@dp.callback_query(F.data.startswith("buy_"))
async def callback_buy(callback: CallbackQuery, state: FSMContext):
    plan_type = callback.data.replace("buy_", "")
    amount    = PLAN_PRICES[plan_type]
    plan_name = PLAN_NAMES[plan_type]  # будет переопределено ниже

    await state.update_data(plan_type=plan_type)
    await state.set_state(PaymentStates.waiting_screenshot)
    await callback.answer()

    # Доллары для каждого тарифа
    lang = await get_user_language(callback.from_user.id)
    caption = get_payment_caption(
        lang=lang,
        plan_name=plan_name,
        amount_som=amount,
        amount_usd=PLAN_PRICES_USD.get(plan_type, 11),
        amount_vnd=PLAN_PRICES_VND.get(plan_type, 280000),
    )

    lang = await get_user_language(callback.from_user.id)
    qr = get_qr_path(lang)
    from texts import TEXTS as _TCK

    # 1. Банковский QR
    if qr.exists():
        try:
            await callback.message.answer_photo(
                photo=FSInputFile(qr),
                caption=caption,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("QR send error: %s", e)
            await callback.message.answer(caption, parse_mode="HTML")
    else:
        await callback.message.answer(caption, parse_mode="HTML")
        qr_upd = _TCK["qr_updating"].get(lang, _TCK["qr_updating"]["ru"])
        await callback.message.answer(qr_upd)

    # 2. USDT блок
    usdt_txt = _TCK["usdt_payment"].get(lang, _TCK["usdt_payment"]["en"])
    usdt_btn_txt = _TCK["usdt_btn"].get(lang, "📱 USDT QR")
    usdt_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=usdt_btn_txt, callback_data="show_usdt_qr")
    ]])
    await callback.message.answer(
        usdt_txt.format(address=USDT_ADDRESS),
        reply_markup=usdt_kb,
        parse_mode="HTML",
    )

    # 3. Visa/MC текст
    _card_txt = _TCK.get("card_coming_soon", {})
    if isinstance(_card_txt, dict):
        _card_txt = _card_txt.get(lang, _card_txt.get("ru", ""))
    if _card_txt:
        await callback.message.answer(_card_txt, parse_mode="HTML")

    await callback.message.answer(_TCK["send_check"].get(lang, _TCK["send_check"]["ru"]))


# ══════════════════════════════════════════════════════════════════════════════
# Приём скриншота
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(PaymentStates.waiting_screenshot, F.photo | F.document)
async def recv_screenshot(message: Message, state: FSMContext):
    data      = await state.get_data()
    plan_type = data.get("plan_type")

    if not plan_type:
        _pchlang = await get_user_language(message.from_user.id)
        from texts import TEXTS as _TPCH
        await message.answer(_TPCH["please_choose_plan"].get(_pchlang, _TPCH["please_choose_plan"]["ru"]))
        await state.clear()
        return

    user      = message.from_user
    amount    = PLAN_PRICES[plan_type]
    plan_name = PLAN_NAMES[plan_type]  # будет переопределено ниже

    file_id = (
        message.photo[-1].file_id if message.photo
        else message.document.file_id
    )

    payment_id = await create_payment(
        telegram_id=user.id,
        plan_type=plan_type,
        screenshot_file_id=file_id,
    )

    # ── OCR анализ чека ───────────────────────────────────────────────────────
    ocr_text = ""
    try:
        # Скачиваем изображение для OCR
        if message.photo:
            file_obj = await bot.get_file(message.photo[-1].file_id)
        else:
            file_obj = await bot.get_file(message.document.file_id)
        img_buf = io.BytesIO()
        await bot.download_file(file_obj.file_path, destination=img_buf)
        img_bytes = img_buf.getvalue()

        ocr_result = await analyze_receipt(img_bytes, amount)
        ocr_text = "\n\n" + format_ocr_result(ocr_result, amount)
    except Exception as e:
        logger.error("OCR failed: %s", e)
        ocr_text = "\n\n🔍 OCR: ошибка анализа"

    username_str  = f"@{user.username}" if user.username else f"id{user.id}"
    admin_caption = (
        f"💳 <b>Новая оплата #{payment_id}</b>\n\n"
        f"👤 {user.first_name} ({username_str})\n"
        f"🆔 <code>{user.id}</code>\n\n"
        f"📅 Тариф: <b>{plan_name}</b>\n"
        f"💰 Сумма: <b>{amount:,} сом</b>\n"
        f"{ocr_text}\n\n"
        "Проверьте перевод и подтвердите 👇"
    )

    admin_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="✅ Подтвердить",
            callback_data=f"approve_{payment_id}_{user.id}_{plan_type}",
        ),
        InlineKeyboardButton(
            text="❌ Отклонить",
            callback_data=f"reject_{payment_id}_{user.id}",
        ),
    ]])

    # ── Сначала отвечаем пользователю — он НЕ видит admin кнопки ─────────────
    await state.clear()
    from texts import TEXTS as _TS
    _slang = await get_user_language(message.from_user.id)
    await message.answer(
        _TS["screenshot_received"].get(_slang, _TS["screenshot_received"]["ru"]),
        parse_mode="HTML",
    )

    # ── Отправляем чек в группу оплат (GROUP_CHAT_ID) ───────────────────────
    # Пользователь этого не видит — он уже получил "Скриншот получен"
    try:
        if message.photo:
            await bot.send_photo(
                chat_id=GROUP_CHAT_ID,
                photo=file_id,
                caption=admin_caption,
                reply_markup=admin_kb,
                parse_mode="HTML",
            )
        else:
            await bot.send_document(
                chat_id=GROUP_CHAT_ID,
                document=file_id,
                caption=admin_caption,
                reply_markup=admin_kb,
                parse_mode="HTML",
            )
        logger.info("Payment #%d forwarded to group %d", payment_id, GROUP_CHAT_ID)
    except Exception as e:
        logger.error("Forward to group failed: %s", e)
        await alert_admin(
            f"Не удалось переслать чек в группу!\n"
            f"Payment #{payment_id} от user {user.id}\n"
            f"Ошибка: {e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Админ: подтверждение / отклонение
# ══════════════════════════════════════════════════════════════════════════════

@dp.callback_query(F.data.startswith("approve_"))
async def callback_approve(callback: CallbackQuery):
    # Только ADMIN_ID может подтверждать — даже в группе
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("⛔️ Только администратор может подтверждать оплату", show_alert=True)
        return

    try:
        parts      = callback.data.split("_", 3)
        payment_id = int(parts[1])
        user_id    = int(parts[2])
        plan_type  = parts[3]
    except Exception:
        await callback.answer("Ошибка данных")
        return

    end_date = await activate_subscription(user_id, plan_type)
    await update_payment_status(payment_id, "approved")

    try:
        from texts import TEXTS as _TAC
        _aclang = await get_user_language(user_id)
        _ac_tpl = _TAC["payment_activated"].get(_aclang, _TAC["payment_activated"]["ru"])
        await bot.send_message(
            chat_id=user_id,
            text=_ac_tpl.format(
                plan=get_plan_name(plan_type, _aclang),
                date=end_date
            ),
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Notify user %d failed: %s", user_id, e)

    try:
        await callback.message.edit_caption(
            caption=callback.message.caption + f"\n\n✅ <b>АКТИВИРОВАНО</b> до {end_date}",
            parse_mode="HTML",
        )
    except Exception:
        pass

    await callback.answer("✅ Подписка активирована")
    logger.info("Approved: payment=%d user=%d plan=%s", payment_id, user_id, plan_type)


@dp.callback_query(F.data.startswith("reject_"))
async def callback_reject(callback: CallbackQuery):
    # Только ADMIN_ID может отклонять — даже в группе
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("⛔️ Только администратор может отклонять оплату", show_alert=True)
        return

    try:
        parts      = callback.data.split("_", 2)
        payment_id = int(parts[1])
        user_id    = int(parts[2])
    except Exception:
        await callback.answer("Ошибка данных")
        return

    await update_payment_status(payment_id, "rejected")

    try:
        from texts import TEXTS as _TRJ
        _rjlang = await get_user_language(user_id)
        await bot.send_message(
            chat_id=user_id,
            text=_TRJ["payment_rejected"].get(_rjlang, _TRJ["payment_rejected"]["ru"]),
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Notify user %d failed: %s", user_id, e)

    try:
        await callback.message.edit_caption(
            caption=callback.message.caption + "\n\n❌ <b>ОТКЛОНЕНО</b>",
            parse_mode="HTML",
        )
    except Exception:
        pass

    await callback.answer("❌ Отклонено")
    logger.info("Rejected: payment=%d user=%d", payment_id, user_id)


# ══════════════════════════════════════════════════════════════════════════════
# Проверка доступа
# ══════════════════════════════════════════════════════════════════════════════

def _limit_exceeded_text() -> str:
    """Единый текст для всех мест где показываем блокировку."""
    return (
        "💎 <b>Бесплатные обработки использованы</b>\n\n"
        "💡 <b>Посчитай сам:</b>\n"
        "Ретушёр берёт 300–1500 сом за одно фото.\n"
        "Retouch Lab — 999 сом в месяц без ограничений.\n\n"
        "⏰ <b>Акция:</b> первый месяц за <b>799 сом</b>\n\n"
        "Выбери тариф 👇"
    )


async def _quick_check(message: Message) -> bool:
    """
    Быстрая предпроверка в handlers — не ставим в очередь
    если лимит ТОЧНО исчерпан прямо сейчас.
    Не атомарная — финальная проверка всё равно в воркере.
    """
    uid = message.from_user.id

    if await check_active_subscription(uid):
        return True

    count = await get_trial_count(uid)
    if count < TRIAL_LIMIT:
        return True

    await log_event(message.from_user.id, "paywall_shown")
    _qlang = await get_user_language(message.from_user.id)
    from texts import TEXTS as _TQC
    await message.answer(
        _TQC["paywall_full"].get(_qlang, _TQC["paywall_full"]["ru"]),
        reply_markup=plans_keyboard(_qlang),
        parse_mode="HTML",
    )
    return False


async def _can_process(message: Message) -> bool:
    """
    Финальная атомарная проверка внутри queue_worker.
    Вызывается непосредственно перед обработкой каждого фото.
    Единая для PHOTO / DOCUMENT / media group / queue.
    """
    uid = message.from_user.id

    # Подписка — всегда пропускаем
    if await check_active_subscription(uid):
        return True

    # Читаем актуальный счётчик из БД прямо перед обработкой
    count = await get_trial_count(uid)
    if count < TRIAL_LIMIT:
        return True

    # Лимит исчерпан
    _clang = await get_user_language(message.from_user.id)
    from texts import TEXTS as _TCP
    await message.answer(
        _TCP["paywall_full"].get(_clang, _TCP["paywall_full"]["ru"]),
        reply_markup=plans_keyboard(_clang),
        parse_mode="HTML",
    )
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Обработка фото
# ══════════════════════════════════════════════════════════════════════════════

async def _process_and_reply(message: Message, data: bytes, filename: str):
    """Ставит фото в очередь пользователя. Обрабатываются по одному."""
    uid = message.from_user.id
    q = _get_queue(uid)
    pos = q.qsize()

    await q.put((data, filename, message))

    if pos > 0:
        await message.answer(
            f"📋 Фото <b>#{pos + 1}</b> добавлено в очередь\n"
            f"⏳ Обработаю после предыдущего",
            parse_mode="HTML",
        )

    # Запускаем воркер если он ещё не запущен для этого пользователя
    if uid not in _queue_active:
        _queue_active.add(uid)
        asyncio.create_task(_queue_worker(uid, bot))


_HEIC_EXTS  = {".heic", ".heif"}
_HEIC_MIMES = {"image/heic", "image/heif", "image/heif-sequence"}
_IMG_MIMES  = {"image/jpeg", "image/png", "image/webp", "image/gif"}


async def log_event(uid: int, event: str):
    """Логируем аналитическое событие в БД."""
    try:
        from database import log_analytics_event
        await log_analytics_event(uid, event)
    except Exception:
        pass  # аналитика не должна ломать основной флоу


async def _do_process(message: Message, data: bytes, filename: str):
    """Внутренняя функция — прямая обработка одного фото."""
    uid = message.from_user.id
    t_start = asyncio.get_event_loop().time()
    logger.info("Recv %.2f MB from %d file=%s", len(data) / 1024 / 1024, uid, filename)

    # Запоминаем до обработки — на подписке или на триале
    has_sub      = bool(await check_active_subscription(uid))
    count_before = await get_trial_count(uid) if not has_sub else 0

    # Логируем событие
    await log_event(uid, "photo_processing_started")

    from texts import TEXTS as _TEXTS
    _ulang = await get_user_language(uid)
    status = await message.answer(_TEXTS["processing_msg"].get(_ulang, "⏳ Обрабатываю фото..."))

    try:
        # Берём пресет режима пользователя
        user_preset = MODES.get(_user_mode.get(uid, DEFAULT_MODE), MODES[DEFAULT_MODE])["preset"]
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_image, data, filename, user_preset)
    except Exception as e:
        logger.error("process_image error: %s", e)
        await status.delete()
        err_str = str(e).lower()
        if "timeout" in err_str or "timed out" in err_str or "read timed" in err_str:
            await message.answer(
                "⏱ <b>Обработка заняла слишком много времени.</b>\n\n"
                "Попробуйте ещё раз или выберите более лёгкий режим:\n"
                "🌿 Натуральная ретушь или ✨ Чистая кожа — работают быстрее.",
                reply_markup=back_to_modes,
                parse_mode="HTML",
            )
        else:
            _errlang = await get_user_language(uid)
        from texts import TEXTS as _TE
        await message.answer(
                _TE["process_error"].get(_errlang, _TE["process_error"]["ru"]),
                reply_markup=make_back_to_modes(_errlang),
            )
        await alert_admin(f"pipeline упал для user={uid}\n{type(e).__name__}: {e}")
        await log_event(uid, "photo_processing_error")
        # НЕ списываем лимит — return до increment_trial_count
        return
    finally:
        pass  # блокировка управляется через _queue_worker

    # Считаем время обработки
    t_total = asyncio.get_event_loop().time() - t_start

    await status.delete()

    stem     = filename.rsplit(".", 1)[0]
    out_name = f"retouched_{stem}.jpg"

    # Размер исходника для отображения
    size_mb = len(data) / 1024 / 1024

    await message.answer_document(BufferedInputFile(result, filename=out_name))
    await log_event(uid, "photo_processed")

    # ── ВАУ-МОМЕНТ — показываем детали обработки ──────────────────────────────
    from texts import TEXTS as _TW
    _wlang = await get_user_language(uid)
    _wow_tpl = _TW["wow_moment"].get(_wlang, _TW["wow_moment"]["ru"])
    wow_text = _wow_tpl.format(
        t=f"{t_total:.0f}",
        size=f"{len(result)/1024/1024:.1f}",
        mins=max(10, int(t_total/3))
    )

    # Считаем trial только после УСПЕШНОЙ обработки и только без подписки
    if not has_sub:
        new_count = await increment_trial_count(uid)
        remaining = TRIAL_LIMIT - new_count

        if remaining <= 0:
            # Последняя бесплатная — показываем продающий экран
            await message.answer(wow_text, parse_mode="HTML")
            await log_event(uid, "paywall_shown")
            from texts import TEXTS as _TW2
            await message.answer(
                _TW2["wow_last_paywall"].get(_wlang, _TW2["wow_last_paywall"]["ru"]),
                reply_markup=plans_keyboard(_wlang),
                parse_mode="HTML",
            )
        elif remaining == 1:
            # Осталась 1 последняя
            await message.answer(
                wow_text + "\n\n" + __import__("texts").TEXTS["trial_last_one"].get(_wlang, __import__("texts").TEXTS["trial_last_one"]["ru"]),
                reply_markup=buy_keyboard(),
                parse_mode="HTML",
            )
        else:
            # Первые фото — показываем вау-момент
            from texts import TEXTS as _TR
            _rlang = await get_user_language(uid)
            _rem_tpl = _TR["trial_remaining"].get(_rlang, _TR["trial_remaining"]["ru"])
            await message.answer(
                wow_text + "\n\n" + _rem_tpl.format(n=remaining),
                parse_mode="HTML",
            )
    else:
        # Подписчик — просто вау-момент без рекламы
        await message.answer(wow_text, parse_mode="HTML")

    logger.info("Done: user=%d size=%.2fMB time=%.1fs",
                uid, len(result) / 1024 / 1024, t_total)

    # Обновляем активность — сбрасываем reminder флаг
    await update_last_active(uid)
    await reset_reminder_flag(uid)


@dp.message(F.document)
async def recv_document(message: Message, state: FSMContext):
    if await state.get_state() == PaymentStates.waiting_screenshot.state:
        await recv_screenshot(message, state)
        return

    doc  = message.document
    mime = (doc.mime_type or "").lower()
    name = doc.file_name or "photo.jpg"
    ext  = ("." + name.rsplit(".", 1)[-1]).lower() if "." in name else ""

    is_heic  = ext in _HEIC_EXTS or mime in _HEIC_MIMES
    is_image = mime in _IMG_MIMES or mime.startswith("image/") or is_heic

    if not is_image:
        await message.answer("Пожалуйста, отправь изображение (JPEG, PNG, HEIC, WebP).")
        return

    # Быстрая предпроверка — не ставим в очередь если лимит точно исчерпан
    if not await _quick_check(message):
        return

    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await _process_and_reply(message, buf.getvalue(), name)


@dp.message(F.photo)
async def recv_photo(message: Message, state: FSMContext):
    # Если ждём скриншот оплаты — принимаем
    if await state.get_state() == PaymentStates.waiting_screenshot.state:
        await recv_screenshot(message, state)
        return

    # Обычное фото — НЕ обрабатываем, НЕ тратим лимит, НЕ ставим в очередь
    from texts import TEXTS as _T
    _bl = await get_user_language(message.from_user.id)
    await message.answer(
        _T["photo_blocked"].get(_bl, _T["photo_blocked"]["ru"]),
        reply_markup=back_menu,
        parse_mode="HTML",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Запуск
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# /admin — список ожидающих платежей (только для админа)
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    if message.from_user.id != ADMIN_ID:
        return

    pending = await get_pending_payments()

    if not pending:
        await message.answer("✅ Ожидающих платежей нет")
        return

    text = f"💳 <b>Ожидают подтверждения: {len(pending)}</b>\n\n"
    for p in pending:
        username_str = f"@{p['username']}" if p.get('username') else f"id{p['telegram_id']}"
        text += (
            f"#{p['id']} — {p['first_name']} ({username_str})\n"
            f"   📅 {PLAN_NAMES.get(p['plan_type'], p['plan_type'])} · "
            f"{p['amount']:,} сом · {p['created_at'][:16]}\n\n"
        )

    await message.answer(text, parse_mode="HTML")


# ══════════════════════════════════════════════════════════════════════════════
# Алерт админу при критической ошибке pipeline
# ══════════════════════════════════════════════════════════════════════════════

async def alert_admin(text: str):
    """Отправляет короткое сообщение админу при критической ошибке."""
    try:
        await bot.send_message(chat_id=ADMIN_ID, text=f"⚠️ <b>Ошибка бота</b>\n\n{text}", parse_mode="HTML")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# BROADCAST — рассылка всем пользователям (только для ADMIN_ID)
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    """Аналитика за 7 дней — только для админа."""
    if message.from_user.id != ADMIN_ID:
        return

    from database import get_analytics_summary
    import sqlite3

    s = await get_analytics_summary()

    # Общая статистика пользователей
    try:
        import aiosqlite
        from database import DB_PATH
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM users") as cur:
                total_users = (await cur.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM subscriptions WHERE is_active=1 AND datetime(end_date) > datetime('now')"
            ) as cur:
                active_subs = (await cur.fetchone())[0]
            async with db.execute(
                "SELECT COUNT(*) FROM users WHERE trial_photos_count > 0"
            ) as cur:
                tried = (await cur.fetchone())[0]
    except Exception:
        total_users = active_subs = tried = 0

    await message.answer(
        f"📊 <b>Аналитика Retouch Lab</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👥 Всего пользователей: <b>{total_users}</b>\n"
        f"🎯 Попробовали trial: <b>{tried}</b>\n"
        f"💎 Активных подписок: <b>{active_subs}</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>За последние 7 дней:</b>\n\n"
        f"📸 Обработано фото: <b>{s.get('photo_processed', 0)}</b>\n"
        f"🚪 Увидели paywall: <b>{s.get('paywall_shown', 0)}</b>\n"
        f"✅ Купили подписку: <b>{s.get('subscriptions_bought', 0)}</b>\n"
        f"📊 Конверсия: <b>{s.get('conversion_pct', 0)}%</b>\n"
        f"❌ Ошибок pipeline: <b>{s.get('photo_processing_error', 0)}</b>",
        parse_mode="HTML",
    )


@dp.message(Command("promo"))
async def cmd_promo(message: Message, state: FSMContext):
    """
    Акционная рассылка — только для администратора.
    Отправляется ТОЛЬКО пользователям БЕЗ активной подписки.
    Использование: /promo
    """
    if message.from_user.id != ADMIN_ID:
        return

    # Кнопка "Купить за 799 сом" — только в акции, не в обычных тарифах
    promo_buy_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="🔥 Купить за 799 сом (~$9)",
            callback_data="buy_promo_1m"
        )
    ]])

    PROMO_TEXTS = {
        "ru": (
            "🔥 ——————————————— 🔥\n"
            "<b>С Ч А С Т Л И В Ы Е</b>\n"
            "<b>      Ч А С Ы</b>\n"
            "🔥 ——————————————— 🔥\n\n"
            "<i>Только сегодня — подписка на 1 месяц</i>\n\n"
            "💎 <b>799 руб</b>  <s>999 руб</s>\n\n"
            "✦ Неограниченная AI-ретушь\n"
            "✦ Оригинальное разрешение 4K / 24MP\n"
            "✦ 5 режимов обработки\n\n"
            "⏰ <b>Акция действует 24 часа</b>"
        ),
        "ky": (
            "🔥 ——————————————— 🔥\n"
            "<b>Б А К Ы Т Т У У</b>\n"
            "<b>С А А Т Т А Р</b>\n"
            "🔥 ——————————————— 🔥\n\n"
            "<i>Бүгүн гана — 1 айлык жазылуу</i>\n\n"
            "💎 <b>799 сом</b>  <s>999 сом</s>\n\n"
            "✦ Чексиз AI ретуши\n"
            "✦ Оригинал сапат 4K / 24MP\n"
            "✦ 5 иштетүү режими\n\n"
            "⏰ <b>Акция 24 саат</b>"
        ),
        "kk": (
            "🔥 ——————————————— 🔥\n"
            "<b>Б А Қ Ы Т Т Ы   С А Ғ А Т Т А Р</b>\n"
            "🔥 ——————————————— 🔥\n\n"
            "<i>Бүгін ғана — 1 айлық жазылым</i>\n\n"
            "💎 <b>4,500 теңге</b>  <s>5,999 теңге</s>\n\n"
            "✦ Шексіз AI ретушь\n"
            "✦ Бастапқы сапат 4K / 24MP\n"
            "✦ 5 өңдеу режимі\n\n"
            "⏰ <b>Акция 24 сағат</b>"
        ),
        "en": (
            "🔥 ——————————————— 🔥\n"
            "<b>H A P P Y   H O U R S</b>\n"
            "🔥 ——————————————— 🔥\n\n"
            "<i>Today only — 1 month subscription</i>\n\n"
            "💎 <b>$9 USDT</b>  <s>$12</s>\n\n"
            "✦ Unlimited AI retouching\n"
            "✦ Original quality 4K / 24MP\n"
            "✦ 5 processing modes\n\n"
            "⏰ <b>Offer valid 24 hours</b>"
        ),
        "vi": (
            "🔥 ——————————————— 🔥\n"
            "<b>G I Ờ   V À N G</b>\n"
            "🔥 ——————————————— 🔥\n\n"
            "<i>Chỉ hôm nay — đăng ký 1 tháng</i>\n\n"
            "💎 <b>250,000 VND</b>  <s>299,000 VND</s>\n\n"
            "✦ Retouch AI không giới hạn\n"
            "✦ Chất lượng gốc 4K / 24MP\n"
            "✦ 5 chế độ xử lý\n\n"
            "⏰ <b>Ưu đãi có hiệu lực 24 giờ</b>"
        ),
    }
    PROMO_TEXT = PROMO_TEXTS.get("ru")  # дефолт для preview

    # Устанавливаем флаг акции на 24 часа
    import time as _time
    global _promo_until
    _promo_until = _time.time() + 86400  # 24 часа

    # Получаем только пользователей БЕЗ подписки
    from database import DB_PATH
    import aiosqlite as _aio
    async with _aio.connect(DB_PATH) as db:
        async with db.execute("""
            SELECT u.telegram_id FROM users u
            WHERE u.is_inactive = 0
            AND NOT EXISTS (
                SELECT 1 FROM subscriptions s
                WHERE s.telegram_id = u.telegram_id
                  AND s.is_active = 1
                  AND datetime(s.end_date) > datetime('now')
            )
        """) as cur:
            rows = await cur.fetchall()
            target_ids = [r[0] for r in rows]

    all_users = await get_all_users()

    confirm_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Отправить", callback_data="promo_send_confirm"),
        InlineKeyboardButton(text="❌ Отмена",    callback_data="promo_cancel"),
    ]])

    await state.update_data(
        promo_text=PROMO_TEXT,
        promo_target_ids=target_ids,
    )
    await state.set_state(BroadcastStates.waiting_confirm)

    await message.answer(
        f"🔥 <b>Акционная рассылка — Счастливые часы</b>\n\n"
        f"Всего пользователей: <b>{len(all_users)}</b>\n"
        f"👥 Получат (без подписки): <b>{len(target_ids)}</b>\n"
        f"💎 Пропущены (есть подписка): <b>{len(all_users) - len(target_ids)}</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{PROMO_TEXT}\n\n"
        f"[кнопка: 🔥 Купить за 799 сом (~$9)]\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Отправить?",
        reply_markup=confirm_kb,
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "promo_send_confirm", BroadcastStates.waiting_confirm)
async def promo_send_confirmed(callback: CallbackQuery, state: FSMContext):
    """Запускаем акционную рассылку только для пользователей без подписки."""
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return

    data = await state.get_data()
    promo_text = data.get("promo_text", "")
    target_ids = data.get("promo_target_ids", [])
    await state.clear()
    await callback.answer()

    promo_buy_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(
            text="🔥 Купить за 799 сом (~$9)",
            callback_data="buy_promo_1m"
        )
    ]])

    status_msg = await callback.message.answer(
        f"🚀 <b>Акционная рассылка запущена...</b>\n"
        f"Получателей: {len(target_ids)}",
        parse_mode="HTML",
    )

    sent = blocked = errors = 0
    from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest

    for i, uid in enumerate(target_ids):
        try:
            user_lang = await get_user_language(uid)
            user_promo = PROMO_TEXTS.get(user_lang, PROMO_TEXTS["ru"])
            await bot.send_message(
                chat_id=uid,
                text=user_promo,
                reply_markup=promo_buy_kb,
                parse_mode="HTML",
            )
            sent += 1
        except TelegramForbiddenError:
            blocked += 1
        except Exception:
            errors += 1

        if (i + 1) % 25 == 0:
            import asyncio as _asyncio
            await _asyncio.sleep(1.5)
        else:
            import asyncio as _asyncio
            await _asyncio.sleep(0.05)

    await status_msg.edit_text(
        f"✅ <b>Акционная рассылка завершена!</b>\n\n"
        f"👥 Отправлено: <b>{len(target_ids)}</b>\n"
        f"✅ Доставлено: <b>{sent}</b>\n"
        f"🚫 Заблокировали: <b>{blocked}</b>\n"
        f"❌ Ошибки: <b>{errors}</b>",
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "promo_cancel", BroadcastStates.waiting_confirm)
async def promo_cancelled(callback: CallbackQuery, state: FSMContext):
    if callback.from_user.id != ADMIN_ID:
        return
    await state.clear()
    await callback.answer()
    await callback.message.edit_text("❌ Рассылка отменена")


@dp.callback_query(F.data == "show_usdt_qr")
async def callback_show_usdt_qr(callback: CallbackQuery):
    """Показывает QR для USDT оплаты."""
    uid = callback.from_user.id
    lang = await get_user_language(uid)
    await callback.answer()
    if USDT_QR_PATH.exists():
        try:
            await callback.message.answer_photo(
                photo=FSInputFile(USDT_QR_PATH),
                caption=f"<code>{USDT_ADDRESS}</code>",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("USDT QR error: %s", e)
            await callback.message.answer(f"<code>{USDT_ADDRESS}</code>", parse_mode="HTML")
    else:
        await callback.message.answer(
            f"<b>USDT {USDT_NETWORK}</b>\n\n<code>{USDT_ADDRESS}</code>",
            parse_mode="HTML",
        )


@dp.message(Command("language"))
async def cmd_language(message: Message, state: FSMContext):
    """Выбор языка — доступно всем пользователям."""
    lang = await get_user_language(message.from_user.id)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🇰🇬 Кыргызча",     callback_data="lang_ky")],
        [InlineKeyboardButton(text="🇷🇺 Русский",      callback_data="lang_ru")],
        [InlineKeyboardButton(text="🇬🇧 English",      callback_data="lang_en")],
        [InlineKeyboardButton(text="🇻🇳 Tiếng Việt",   callback_data="lang_vi")],
    ])
    await message.answer(
        "🌐 <b>Выберите язык / Choose language / Chọn ngôn ngữ / Тилди тандаңыз:</b>",
        reply_markup=kb,
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("lang_"))
async def callback_set_language(callback: CallbackQuery):
    """Устанавливает язык и сразу показывает главное меню."""
    uid = callback.from_user.id
    lang = callback.data.replace("lang_", "")

    if lang not in LANGUAGES:
        await callback.answer()
        return

    await set_user_language(uid, lang)
    await callback.answer()

    # Закрываем инлайн меню
    try:
        await callback.message.delete()
    except Exception:
        pass

    # Строим меню на новом языке
    has_sub = bool(await check_active_subscription(uid))
    trial_used = await get_trial_count(uid)
    remaining = max(0, TRIAL_LIMIT - trial_used)
    import time as _time
    promo_active = _promo_until > _time.time()
    new_menu = make_main_menu(lang, remaining, has_sub, promo_active)

    # Баннер если есть
    banner_path = Path(__file__).parent / "start_banner.png"
    if banner_path.exists():
        try:
            await callback.message.answer_photo(
                photo=FSInputFile(str(banner_path)),
                caption="✨ <b>RETOUCH LAB</b>",
                parse_mode="HTML",
            )
        except Exception:
            pass

    # Сразу главное меню на новом языке
    from texts import TEXTS
    await callback.message.answer(
        TEXTS["welcome"].get(lang, TEXTS["welcome"]["ru"]),
        reply_markup=new_menu,
        parse_mode="HTML",
    )


@dp.message(Command("relaunch"))
async def cmd_relaunch(message: Message):
    """Отправляет сообщение о возвращении бота всем пользователям."""
    if message.from_user.id != ADMIN_ID:
        return
    users = await get_all_users()
    status_msg = await message.answer(
        f"🚀 Отправляем relaunch message {len(users)} пользователям...",
    )
    stats = await broadcast(bot, RELAUNCH_MESSAGE)
    await status_msg.edit_text(
        f"✅ <b>Relaunch рассылка завершена!</b>\n\n"
        f"👥 Всего: {stats['total']}\n"
        f"✅ Доставлено: {stats['sent']}\n"
        f"🚫 Заблокировали: {stats['blocked']}",
        parse_mode="HTML",
    )


@dp.message(Command("broadcast"))
async def cmd_broadcast(message: Message, state: FSMContext):
    """Начинает flow рассылки. Только для администратора."""
    if message.from_user.id != ADMIN_ID:
        return

    users = await get_all_users()
    await state.set_state(BroadcastStates.waiting_text)

    await message.answer(
        f"📢 <b>Рассылка</b>\n\n"
        f"Активных пользователей: <b>{len(users)}</b>\n\n"
        f"Отправьте текст сообщения.\n"
        f"Поддерживается HTML-форматирование:\n"
        f"<b>жирный</b>, <i>курсив</i>, <code>моноширинный</code>\n\n"
        f"Для отмены: /cancel",
        parse_mode="HTML",
    )


@dp.message(Command("cancel"), BroadcastStates.waiting_text)
@dp.message(Command("cancel"), BroadcastStates.waiting_confirm)
async def cmd_cancel_broadcast(message: Message, state: FSMContext):
    if message.from_user.id != ADMIN_ID:
        return
    await state.clear()
    await message.answer("❌ Рассылка отменена")


@dp.message(BroadcastStates.waiting_text)
async def broadcast_got_text(message: Message, state: FSMContext):
    """Получили текст — показываем preview и просим подтверждение."""
    if message.from_user.id != ADMIN_ID:
        return

    text = message.text or message.caption or ""
    if not text:
        await message.answer("Отправьте текстовое сообщение")
        return

    await state.update_data(broadcast_text=text)
    await state.set_state(BroadcastStates.waiting_confirm)

    users = await get_all_users()

    confirm_kb = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Отправить всем", callback_data="broadcast_confirm"),
        InlineKeyboardButton(text="❌ Отмена",         callback_data="broadcast_cancel"),
    ]])

    await message.answer(
        f"👁 <b>Preview сообщения:</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{text}\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Будет отправлено: <b>{len(users)}</b> пользователям\n\n"
        f"Подтвердите отправку:",
        reply_markup=confirm_kb,
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "broadcast_confirm", BroadcastStates.waiting_confirm)
async def broadcast_confirmed(callback: CallbackQuery, state: FSMContext):
    """Подтверждение — запускаем рассылку."""
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return

    data = await state.get_data()
    text = data.get("broadcast_text", "")
    await state.clear()
    await callback.answer()

    status_msg = await callback.message.answer(
        "🚀 <b>Рассылка запущена...</b>\n\nЭто займёт некоторое время.",
        parse_mode="HTML",
    )

    async def on_progress(done, total, sent, blocked, errors):
        try:
            await status_msg.edit_text(
                f"🚀 <b>Рассылка идёт...</b>\n\n"
                f"📤 Отправлено: {done}/{total}\n"
                f"✅ Доставлено: {sent}\n"
                f"🚫 Заблокировали: {blocked}\n"
                f"❌ Ошибки: {errors}",
                parse_mode="HTML",
            )
        except Exception:
            pass

    stats = await broadcast(bot, text, progress_callback=on_progress)

    await status_msg.edit_text(
        f"✅ <b>Рассылка завершена!</b>\n\n"
        f"👥 Всего: {stats['total']}\n"
        f"✅ Доставлено: {stats['sent']}\n"
        f"🚫 Заблокировали бота: {stats['blocked']}\n"
        f"❌ Ошибки: {stats['errors']}",
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "broadcast_cancel", BroadcastStates.waiting_confirm)
async def broadcast_cancelled(callback: CallbackQuery, state: FSMContext):
    if callback.from_user.id != ADMIN_ID:
        return
    await state.clear()
    await callback.answer()
    await callback.message.edit_text("❌ Рассылка отменена")


async def main():
    logger.info("Retouch Lab v3 starting...")
    await init_db()

    scheduler = AsyncIOScheduler()
    # Уведомления об окончании подписки — каждый день в 10:00
    scheduler.add_job(
        send_expiry_notifications,
        trigger="cron",
        hour=10,
        minute=0,
        args=[bot],
    )
    # Reminder неактивным пользователям — каждый день в 12:00
    scheduler.add_job(
        send_reminders,
        trigger="cron",
        hour=12,
        minute=0,
        args=[bot],
    )
    scheduler.start()
    logger.info("Scheduler started — daily notifications at 10:00")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
