"""
bot.py — Telegram бот Retouch Lab (aiogram 3.x)

Принимает: photo, document (JPEG/PNG/HEIC/WebP)
Отправляет: ТОЛЬКО sendDocument, имя retouched_<original>.jpg
"""

import asyncio
import io
import logging
import os

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    BufferedInputFile,
)
from aiogram.filters import Command
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
RUNPOD_URL = os.getenv(
    "RUNPOD_URL",
    "https://ychsinzriqqmll-8000.proxy.runpod.net/process-image",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ── Клавиатуры ─────────────────────────────────────────────────────────────────

main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📸 Обработать фото")],
        [KeyboardButton(text="✨ Улучшить качество")],
        [KeyboardButton(text="💎 Подписка")],
        [KeyboardButton(text="ℹ️ О боте")],
    ],
    resize_keyboard=True,
)

back_menu = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="⬅️ Назад")]],
    resize_keyboard=True,
)

# ── Хендлеры меню ──────────────────────────────────────────────────────────────

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Привет! 👋\n\n"
        "Я *Retouch Lab* — AI ретушёр от *Linkiway*.\n\n"
        "Что я умею:\n\n"
        "✨ убираю дефекты кожи\n"
        "🎨 сохраняю натуральный тон\n"
        "🧴 natural skin tone\n"
        "📷 сохраняю оригинальное качество\n\n"
        "Отправьте фото — как фотографию или файлом 👇",
        reply_markup=main_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "📸 Обработать фото")
async def process_photo_menu(message: Message):
    await message.answer(
        "📸 *Обработка фото*\n\n"
        "Отправьте фото *файлом* для максимального качества\n"
        "или просто как фотографию.\n\n"
        "Поддерживаю:\n"
        "• JPEG / PNG\n"
        "• iPhone (HEIC)\n"
        "• Android фото\n\n"
        "Что сделаю:\n\n"
        "✨ уберу дефекты кожи\n"
        "🎨 сохраню натуральный тон\n"
        "🧴 аккуратная ретушь без пластика\n"
        "📷 оригинальное разрешение и качество\n\n"
        "Жду фото 📸",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "✨ Улучшить качество")
async def enhance_menu(message: Message):
    await message.answer(
        "✨ *Улучшение качества*\n\n"
        "AI аккуратно улучшит:\n\n"
        "• чистоту кожи\n"
        "• детализацию\n"
        "• локальный контраст\n\n"
        "Отправьте фото файлом или как фото.",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "💎 Подписка")
async def subscription(message: Message):
    await message.answer(
        "💎 *Подписка Retouch Lab*\n\n"
        "Выберите удобный тариф:\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📅 *1 месяц*\n"
        "20$ / ~1 800 сом\n"
        "До 300 фото\n\n"
        "📅 *3 месяца*\n"
        "50$ / ~4 500 сом\n"
        "Экономия 17%\n\n"
        "📅 *6 месяцев*\n"
        "80$ / ~7 000 сом\n"
        "Экономия 33%\n\n"
        "📅 *1 год*\n"
        "115$ / ~10 000 сом\n"
        "Экономия 52% 🔥\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Все тарифы включают:\n"
        "✅ AI ретушь кожи\n"
        "✅ Сохранение оригинального качества\n"
        "✅ Natural skin tone\n\n"
        "💍 *Скоро* — эксклюзивная функция для свадебных фотографов!\n"
        "Пакетная обработка галереи с единым стилем ретуши.\n\n"
        "По вопросам оплаты: @linkiway\\_support",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "ℹ️ О боте")
async def about(message: Message):
    await message.answer(
        "ℹ️ *О Retouch Lab*\n\n"
        "Retouch Lab — AI ретушёр от *Linkiway*.\n\n"
        "Мы создали его чтобы сэкономить время:\n\n"
        "📸 фотографам\n"
        "🎥 блогерам\n"
        "💍 свадебным фотографам\n"
        "🧑‍💻 креаторам\n\n"
        "AI делает:\n\n"
        "✨ убирает дефекты кожи\n"
        "🎨 сохраняет натуральный тон\n"
        "🧴 natural skin tone без пластика\n"
        "📷 оригинальное разрешение и качество\n\n"
        "То, что раньше занимало 20 минут ретуши — теперь секунды.\n\n"
        "🔗 *Linkiway* — технологии для творческих людей.",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "⬅️ Назад")
async def back(message: Message):
    await message.answer("Главное меню 👇", reply_markup=main_menu)


# ── Общая функция отправки в API ───────────────────────────────────────────────

async def _send_to_api(
    image_bytes: bytes,
    user_id: str,
    filename: str,
) -> bytes | str:
    """
    Отправляет изображение в FastAPI backend.
    Возвращает байты результата или строку-статус ('limit', 'timeout', 'error').
    """
    try:
        timeout = aiohttp.ClientTimeout(total=180)  # 3 минуты
        async with aiohttp.ClientSession(timeout=timeout) as session:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                image_bytes,
                filename=filename,
                content_type="image/jpeg",
            )
            form.add_field("user_id", user_id)

            async with session.post(RUNPOD_URL, data=form) as resp:
                if resp.status == 429:
                    return "limit"
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("API error %d: %s", resp.status, text[:200])
                    return "error"
                return await resp.read()

    except asyncio.TimeoutError:
        logger.error("API timeout")
        return "timeout"
    except Exception as exc:
        logger.exception("API request failed: %s", exc)
        return "error"


async def _process_and_reply(
    message: Message,
    image_bytes: bytes,
    original_filename: str,
):
    """Скачал → API → отправить файлом."""
    user_id = str(message.from_user.id)
    mb = len(image_bytes) / 1024 / 1024
    logger.info("Received %.2f MB from user %s, file=%s", mb, user_id, original_filename)

    status_msg = await message.answer("⏳ Обрабатываю фото, подождите...")

    result = await _send_to_api(image_bytes, user_id, original_filename)

    await status_msg.delete()

    if result == "limit":
        await message.answer("❌ Лимит обработок исчерпан.")
        return
    if result == "timeout":
        await message.answer("❌ Сервер не ответил. Попробуйте через минуту.")
        return
    if result == "error" or not isinstance(result, bytes):
        await message.answer("❌ Ошибка обработки. Попробуйте ещё раз.")
        return

    # Имя выходного файла: retouched_<original>.jpg
    stem = original_filename.rsplit(".", 1)[0]
    out_filename = f"retouched_{stem}.jpg"

    # ТОЛЬКО sendDocument — сохраняет оригинальное качество
    output_file = BufferedInputFile(result, filename=out_filename)
    await message.answer_document(
        output_file,
        caption=(
            "✨ *Готово!*\n\n"
            "Файл сохранён в оригинальном качестве.\n"
            f"📄 `{out_filename}`"
        ),
        parse_mode="Markdown",
    )

    result_mb = len(result) / 1024 / 1024
    logger.info("Sent result %.2f MB to user %s", result_mb, user_id)


# ── Приём фото как ДОКУМЕНТ (несжатое — лучшее качество) ──────────────────────

@dp.message(F.document)
async def get_photo_as_document(message: Message):
    doc = message.document

    # Принимаем только изображения
    allowed_mime = {"image/jpeg", "image/png", "image/heic", "image/heif", "image/webp"}
    if doc.mime_type and doc.mime_type not in allowed_mime:
        if not doc.mime_type.startswith("image/"):
            await message.answer(
                "Пожалуйста, отправьте изображение (JPEG, PNG, HEIC, WebP)."
            )
            return

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    filename = doc.file_name or "photo.jpg"
    await _process_and_reply(message, image_bytes, filename)


# ── Приём фото как ФОТО (Telegram сжимает до ~1280px) ─────────────────────────

@dp.message(F.photo)
async def get_photo_as_photo(message: Message):
    # Самое большое доступное разрешение
    photo = message.photo[-1]

    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    await message.answer(
        "💡 *Совет:* для лучшего качества отправляйте фото *файлом*\n"
        "(скрепка → Файл → выберите фото)",
        parse_mode="Markdown",
    )
    await _process_and_reply(message, image_bytes, "photo.jpg")


# ── Запуск ─────────────────────────────────────────────────────────────────────

async def main():
    logger.info("Retouch Lab bot starting…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
