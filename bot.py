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
    PLAN_PRICES,
    TRIAL_LIMIT,
)
from notifications import send_expiry_notifications
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
    "✦ <b>Retouch Lab</b> — AI-ретушь для всех\n\n"

    "Профессиональная Dodge & Burn ретушь за секунды.\n"
    "Без потери качества.\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "✨ <b>Для обычных пользователей и блогеров</b>\n"
    "✦ Clean skin без plastic-эффекта\n"
    "✦ Улучши селфи и повседневные фото\n"
    "✦ Удобно и просто — просто отправь фото\n"
    "✦ Моментальный результат прямо в Telegram\n"
    "✦ Не нужен опыт — справится каждый\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "📷 <b>Для фотографов</b>\n"
    "✦ Оригинальное разрешение — 4K / 24MP сохраняется\n"
    "✦ JPEG quality 97, без сильного сжатия\n"
    "✦ Родинки, текстура и поры остаются\n"
    "✦ Natural retouch — готово для клиентской отдачи\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "🎁 <b>3 бесплатные обработки</b> для новых пользователей\n\n"

    "Отправь фото <b>файлом</b> (📎 → Файл) для полного качества\n"
    "Форматы: JPG · PNG · HEIC · WebP"
)

# О нас / About
ABOUT_TEXT = (
    "✦ <b>Retouch Lab</b> — профессиональный AI-ретушёр от Linkiway\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "🎯 <b>Наш подход</b>\n\n"
    "Не AI beauty filter.\n"
    "А <b>натуральная Dodge & Burn ретушь</b> — "
    "как делают топовые ретушёры вручную.\n\n"

    "✦ Clean skin — без мыльного эффекта\n"
    "✦ Текстура кожи и поры сохраняются\n"
    "✦ Родинки и особенности лица остаются\n"
    "✦ Натуральный тёплый skin tone\n"
    "✦ Polished expensive look\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "📐 <b>Качество файла</b>\n\n"
    "✦ Оригинальное разрешение сохраняется\n"
    "✦ Поддержка 4K / 24MP и выше\n"
    "✦ Не уменьшает фото до web-preview\n"
    "✦ Подходит для профессиональной печати\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "⚡ <b>Экономия времени</b>\n\n"
    "20–40 минут ручной ретуши → секунды.\n"
    "Без потери качества.\n\n"

    "━━━━━━━━━━━━━━━━━━\n"
    "🚀 Retouch Lab постоянно развивается.\n"
    "В будущем — новые AI-инструменты и функции."
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
# GROUP_CHAT_ID — отдельная группа куда приходят чеки с кнопками approve/reject
# Получить: создай группу → добавь бота → напиши /start → смотри getUpdates
GROUP_CHAT_ID  = int(os.getenv("GROUP_CHAT_ID", os.getenv("ADMIN_CHAT_ID", "532189427")))
QR_PATH        = Path(__file__).parent / "qr_code.png"
MBANK_PHONE    = "+996 (500) 070 759"
MBANK_NAME     = "АЛИНА А."

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
class PaymentStates(StatesGroup):
    waiting_screenshot = State()


class BroadcastStates(StatesGroup):
    waiting_text    = State()   # ждём текст рассылки
    waiting_confirm = State()   # ждём подтверждение


# ── Клавиатуры ────────────────────────────────────────────────────────────────
main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📸 Обработать фото")],
        [KeyboardButton(text="💎 Подписка"), KeyboardButton(text="🖼 Before / After")],
        [KeyboardButton(text="📂 Форматы"), KeyboardButton(text="ℹ️ О нас")],
        [KeyboardButton(text="💬 Поддержка")],
    ],
    resize_keyboard=True,
)

back_menu = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="⬅️ Назад")]],
    resize_keyboard=True,
)


def plans_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=f"📅 1 месяц — {PLAN_PRICES['1m']:,} сом (~$11)",
            callback_data="buy_1m",
        )],
        [InlineKeyboardButton(
            text=f"📅 3 месяца — {PLAN_PRICES['3m']:,} сом (~$28) · -15%",
            callback_data="buy_3m",
        )],
        [InlineKeyboardButton(
            text=f"📅 6 месяцев — {PLAN_PRICES['6m']:,} сом (~$57) · -25%",
            callback_data="buy_6m",
        )],
        [InlineKeyboardButton(
            text=f"📅 1 год — {PLAN_PRICES['1y']:,} сом (~$102) · -35% 🔥",
            callback_data="buy_1y",
        )],
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
    await message.answer(
        WELCOME_TEXT,
        reply_markup=main_menu,
        parse_mode="HTML",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Меню
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(F.text == "⬅️ Назад")
async def back(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Главное меню 👇", reply_markup=main_menu)


@dp.message(F.text == "ℹ️ О нас")
async def menu_about(message: Message):
    await message.answer(
        ABOUT_TEXT,
        reply_markup=back_menu,
        parse_mode="HTML",
    )


@dp.message(F.text == "💬 Поддержка")
async def menu_support(message: Message):
    await message.answer(
        SUPPORT_TEXT,
        reply_markup=back_menu,
        parse_mode="HTML",
    )


@dp.message(F.text == "🖼 Before / After")
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

    DESCRIPTION = (
        "✨ <b>Примеры нашей ретуши:</b>\n\n"
        "• Натуральное выравнивание кожи\n"
        "• Текстура и поры сохраняются\n"
        "• Родинки и особенности остаются\n"
        "• Полированный дорогой результат\n\n"
        "Попробуй прямо сейчас — отправь своё фото ✨"
    )

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
            await message.answer("👆", reply_markup=back_menu)
        except Exception as e:
            logger.error("[BEFORE_AFTER] photo send failed: %s", e)
            await message.answer(DESCRIPTION, reply_markup=back_menu, parse_mode="HTML")
        return

    # ── Fallback: нет ни видео ни фото ───────────────────────────────────────
    await message.answer(
        "🖼 <b>Before / After — Retouch Lab</b>\n\n"
        + DESCRIPTION + "\n\n"
        "Скоро здесь появятся примеры 📸",
        reply_markup=back_menu,
        parse_mode="HTML",
    )


@dp.message(F.text == "📂 Форматы")
async def menu_formats(message: Message):
    await message.answer(
        "📂 <b>Поддерживаемые форматы</b>\n\n"
        "✦ <b>JPEG / JPG</b> — любые камеры\n"
        "✦ <b>PNG</b> — без потерь\n"
        "✦ <b>HEIC / HEIF</b> — iPhone / iPad\n"
        "✦ <b>WebP</b> — веб-форматы\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📐 <b>Разрешение</b>\n\n"
        "✦ Поддержка до 4K и выше\n"
        "✦ 24MP, 36MP, 50MP — всё принимается\n"
        "✦ Оригинальный размер сохраняется\n"
        "✦ Aspect ratio не изменяется\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "⚠️ <b>Важно</b>\n\n"
        "Отправляй фото <b>файлом</b>, а не через галерею.\n"
        "Telegram сжимает обычные фото до ~1200px.\n\n"
        "📎 Скрепка → <b>Файл</b> → выбери фото",
        reply_markup=back_menu,
        parse_mode="HTML",
    )


@dp.message(F.text == "📸 Обработать фото")
async def menu_process(message: Message):
    await message.answer(
        SEND_PHOTO_TEXT,
        reply_markup=back_menu,
        parse_mode="HTML",
    )


@dp.message(F.text == "💎 Подписка")
async def menu_sub(message: Message):
    sub = await check_active_subscription(message.from_user.id)

    if sub:
        end_pretty = ".".join(reversed(sub["end_date"][:10].split("-")))
        await message.answer(
            f"💎 <b>Подписка активна</b>\n\n"
            f"📅 Тариф: <b>{PLAN_NAMES.get(sub['plan_type'], sub['plan_type'])}</b>\n"
            f"⏳ Действует до: <b>{end_pretty}</b>\n\n"
            "Отправляйте фото — всё работает ✨",
            reply_markup=back_menu,
            parse_mode="HTML",
        )
        return

    await message.answer(
        "💎 <b>Подписка Retouch Lab</b>\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 1 месяц — <b>990 сом</b> (~$11)\n"
        f"📅 3 месяца — <b>2 490 сом</b> (~$28) · -15%\n"
        f"📅 6 месяцев — <b>4 990 сом</b> (~$57) · -25%\n"
        f"📅 1 год — <b>8 990 сом</b> (~$102) · -35% 🔥\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Выберите тариф 👇",
        reply_markup=plans_keyboard(),
        parse_mode="HTML",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Выбор тарифа → реквизиты
# ══════════════════════════════════════════════════════════════════════════════

@dp.callback_query(F.data == "open_plans")
async def callback_open_plans(callback: CallbackQuery):
    await callback.answer()
    await callback.message.answer(
        "💎 <b>Подписка Retouch Lab</b>\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📅 1 месяц — <b>990 сом</b> (~$11)\n"
        f"📅 3 месяца — <b>2 490 сом</b> (~$28) · -15%\n"
        f"📅 6 месяцев — <b>4 990 сом</b> (~$57) · -25%\n"
        f"📅 1 год — <b>8 990 сом</b> (~$102) · -35% 🔥\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Выберите тариф 👇",
        reply_markup=plans_keyboard(),
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("buy_"))
async def callback_buy(callback: CallbackQuery, state: FSMContext):
    plan_type = callback.data.replace("buy_", "")
    amount    = PLAN_PRICES[plan_type]
    plan_name = PLAN_NAMES[plan_type]

    await state.update_data(plan_type=plan_type)
    await state.set_state(PaymentStates.waiting_screenshot)
    await callback.answer()

    # Доллары для каждого тарифа
    usd_map = {"1m": "$11", "3m": "$28", "6m": "$57", "1y": "$102"}
    usd = usd_map.get(plan_type, "")

    caption = (
        f"💳 <b>Оплата: {plan_name} — {amount:,} сом ({usd})</b>\n\n"
        f"Переведите <b>{amount:,} сом</b> на MBank:\n\n"
        f"👤 <b>{MBANK_NAME}</b>\n"
        f"📱 <b>{MBANK_PHONE}</b>\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📌 <b>Как оплатить:</b>\n"
        "1. Откройте MBank\n"
        "2. Переводы → По номеру телефона\n"
        "   или отсканируйте QR-код\n"
        f"3. Введите сумму: <b>{amount:,} сом ({usd})</b>\n"
        "4. Переведите и сохраните чек\n"
        "5. Отправьте скриншот сюда 👇\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "⏳ Подписка активируется в течение нескольких минут."
    )

    if QR_PATH.exists():
        await callback.message.answer_photo(
            photo=FSInputFile(QR_PATH),
            caption=caption,
            parse_mode="HTML",
        )
    else:
        await callback.message.answer(caption, parse_mode="HTML")

    await callback.message.answer("📸 Отправьте скриншот чека 👇")


# ══════════════════════════════════════════════════════════════════════════════
# Приём скриншота
# ══════════════════════════════════════════════════════════════════════════════

@dp.message(PaymentStates.waiting_screenshot, F.photo | F.document)
async def recv_screenshot(message: Message, state: FSMContext):
    data      = await state.get_data()
    plan_type = data.get("plan_type")

    if not plan_type:
        await message.answer("Пожалуйста, выберите тариф через меню 💎 Подписка")
        await state.clear()
        return

    user      = message.from_user
    amount    = PLAN_PRICES[plan_type]
    plan_name = PLAN_NAMES[plan_type]

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
    await message.answer(
        "✅ <b>Скриншот получен!</b>\n\n"
        "Проверяем оплату — подписка активируется\n"
        "в течение нескольких минут 💎",
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
        await bot.send_message(
            chat_id=user_id,
            text=(
                f"✅ <b>Подписка успешно активирована!</b>\n\n"
                f"📅 Тариф: <b>{PLAN_NAMES.get(plan_type, plan_type)}</b>\n"
                f"⏳ Действует до: <b>{end_date}</b>\n\n"
                "Отправляйте фото для ретуши 📸✨"
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
        await bot.send_message(
            chat_id=user_id,
            text=(
                "❌ <b>Платёж отклонён.</b>\n\n"
                "Возможные причины:\n"
                "• Сумма не совпадает\n"
                "• Перевод не найден\n"
                "• Нечёткий скриншот\n\n"
                "Свяжитесь с поддержкой: @linkiway_support"
            ),
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
        "Retouch Lab — 990 сом в месяц без ограничений.\n\n"
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
    await message.answer(
        _limit_exceeded_text(),
        reply_markup=plans_keyboard(),
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
    await message.answer(
        _limit_exceeded_text(),
        reply_markup=plans_keyboard(),
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

    status = await message.answer("⏳ Обрабатываю фото...")

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_image, data, filename)
    except Exception as e:
        logger.error("process_image error: %s", e)
        await status.delete()
        await message.answer("❌ Ошибка обработки. Попробуй ещё раз.")
        await alert_admin(f"pipeline упал для user={uid}\n{type(e).__name__}: {e}")
        await log_event(uid, "photo_processing_error")
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
    wow_text = (
        f"✅ <b>Готово!</b> Время обработки: <b>{t_total:.0f} сек</b>\n\n"
        f"📐 Разрешение: оригинал сохранён\n"
        f"🔍 Текстура кожи: сохранена\n"
        f"✦ Родинки и особенности: на месте\n"
        f"💾 Размер файла: {len(result)/1024/1024:.1f} MB\n\n"
        f"<i>Ретушёр делал бы это {max(10, int(t_total/3))}–30 минут вручную.</i>"
    )

    # Считаем trial только после УСПЕШНОЙ обработки и только без подписки
    if not has_sub:
        new_count = await increment_trial_count(uid)
        remaining = TRIAL_LIMIT - new_count

        if remaining <= 0:
            # Последняя бесплатная — показываем продающий экран
            await message.answer(wow_text, parse_mode="HTML")
            await log_event(uid, "paywall_shown")
            await message.answer(
                "💎 <b>Бесплатные обработки использованы</b>\n\n"
                "Ты уже видел результат — натуральная ретушь\n"
                "без потери качества и без пластика.\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "💡 <b>Посчитай сам:</b>\n\n"
                "Ретушёр берёт <b>300–1500 сом</b> за одно фото.\n"
                "Retouch Lab — <b>990 сом в месяц</b> без ограничений.\n\n"
                "Это окупается с первого же фото. 🎯\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🚀 <b>Ты один из первых 100 пользователей</b>\n"
                "Ранние пользователи получают лучшую цену.\n\n"
                "⏰ <b>Акция 48 часов:</b> месяц за <b>799 сом</b> вместо 990\n\n"
                "Выбери тариф 👇",
                reply_markup=plans_keyboard(),
                parse_mode="HTML",
            )
        elif remaining == 1:
            # Осталась 1 последняя
            await message.answer(
                wow_text + "\n\n"
                "⚠️ <b>Осталась 1 бесплатная обработка</b>\n\n"
                "После неё потребуется подписка.\n"
                "Оформи заранее чтобы не прерываться 👇",
                reply_markup=buy_keyboard(),
                parse_mode="HTML",
            )
        else:
            # Первые фото — показываем вау-момент
            await message.answer(
                wow_text + f"\n\n🎁 Осталось бесплатных: <b>{remaining}</b>",
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
    await message.answer(
        PHOTO_BLOCKED_TEXT,
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