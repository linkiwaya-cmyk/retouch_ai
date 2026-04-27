"""
bot.py — Retouch Lab (aiogram 3.x)

Изменения:
- Caption: только "Готово ✨" — без лишнего текста
- Принимает JPEG / PNG / HEIC (iPhone) / WebP
- sendDocument только (не sendPhoto)
- filename: retouched_<original>.jpg
"""
import asyncio
import io
import logging
import os

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, BufferedInputFile
from aiogram.filters import Command
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN  = os.getenv("BOT_TOKEN")
RUNPOD_URL = os.getenv("RUNPOD_URL", "http://localhost:8000/process-image")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher()

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

# ── Меню ───────────────────────────────────────────────────────────────────────

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Привет! 👋\n\n"
        "Я *Retouch Lab* — AI ретушёр от *Linkiway*.\n\n"
        "Отправьте фото и я сделаю профессиональную ретушь:\n\n"
        "✨ уберу дефекты кожи\n"
        "🎨 выровняю тон\n"
        "🧴 сохраню текстуру и натуральность\n"
        "📷 верну файл в оригинальном качестве\n\n"
        "Поддерживаю: JPEG, PNG, HEIC (iPhone), WebP",
        reply_markup=main_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "📸 Обработать фото")
async def menu_process(message: Message):
    await message.answer(
        "📸 Отправьте фото *файлом* для максимального качества\n"
        "или просто как фотографию.\n\n"
        "iPhone (HEIC), Android (JPEG/PNG) — всё принимается 👇",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "✨ Улучшить качество")
async def menu_enhance(message: Message):
    await message.answer(
        "✨ Отправьте фото — я улучшу:\n\n"
        "• чистоту кожи\n"
        "• выравниваю тон\n"
        "• аккуратный Dodge & Burn",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "💎 Подписка")
async def menu_sub(message: Message):
    await message.answer(
        "💎 *Подписка Retouch Lab*\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📅 *1 месяц*\n"
        "20$ / ~1 800 сом · до 300 фото\n\n"
        "📅 *3 месяца*\n"
        "50$ / ~4 500 сом · экономия 17%\n\n"
        "📅 *6 месяцев*\n"
        "80$ / ~7 000 сом · экономия 33%\n\n"
        "📅 *1 год*\n"
        "115$ / ~10 000 сом · экономия 52% 🔥\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "✅ AI ретушь кожи\n"
        "✅ Оригинальное качество\n"
        "✅ Natural skin tone\n\n"
        "💍 *Скоро* — функция для свадебных фотографов!\n\n"
        "По вопросам: @linkiway\\_support",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "ℹ️ О боте")
async def menu_about(message: Message):
    await message.answer(
        "ℹ️ *Retouch Lab* — AI ретушёр от *Linkiway*\n\n"
        "📸 фотографам · 🎥 блогерам · 💍 свадебным фотографам\n\n"
        "То, что раньше занимало 20 минут — теперь секунды.\n\n"
        "🔗 *Linkiway* — технологии для творческих людей.",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "⬅️ Назад")
async def back(message: Message):
    await message.answer("Главное меню 👇", reply_markup=main_menu)


# ── API ────────────────────────────────────────────────────────────────────────

async def _send_to_api(image_bytes: bytes, user_id: str, filename: str) -> bytes | str:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
            form = aiohttp.FormData()
            form.add_field("file", image_bytes, filename=filename, content_type="image/jpeg")
            form.add_field("user_id", user_id)
            async with s.post(RUNPOD_URL, data=form) as r:
                if r.status == 429:
                    return "limit"
                if r.status != 200:
                    text = await r.text()
                    logger.error("API %d: %s", r.status, text[:200])
                    return "error"
                return await r.read()
    except asyncio.TimeoutError:
        return "timeout"
    except Exception as e:
        logger.exception("API error: %s", e)
        return "error"


async def _process_and_reply(message: Message, image_bytes: bytes, original_filename: str):
    user_id = str(message.from_user.id)
    mb = len(image_bytes) / 1024 / 1024
    logger.info("Received %.2f MB from %s, file=%s", mb, user_id, original_filename)

    status = await message.answer("⏳ Обрабатываю…")
    result = await _send_to_api(image_bytes, user_id, original_filename)
    await status.delete()

    if result == "limit":
        await message.answer("❌ Лимит обработок исчерпан.")
        return
    if result == "timeout":
        await message.answer("❌ Сервер не ответил. Попробуйте позже.")
        return
    if result == "error" or not isinstance(result, bytes):
        await message.answer("❌ Ошибка обработки. Попробуйте ещё раз.")
        return

    stem = original_filename.rsplit(".", 1)[0]
    out_name = f"retouched_{stem}.jpg"

    # Только "Готово ✨" — без лишнего текста
    await message.answer_document(
        BufferedInputFile(result, filename=out_name),
        caption="Готово ✨",
    )
    logger.info("Sent %.2f MB to %s as %s", len(result)/1024/1024, user_id, out_name)


# ── Приём документов (HEIC, JPEG, PNG, WebP) ───────────────────────────────────

@dp.message(F.document)
async def recv_document(message: Message):
    doc = message.document
    mime = doc.mime_type or ""

    # Принимаем все image/* + HEIC специально
    heic_names = {".heic", ".heif"}
    ext = ("." + (doc.file_name or "").rsplit(".", 1)[-1]).lower()
    is_heic = ext in heic_names or mime in {"image/heic", "image/heif"}

    if not mime.startswith("image/") and not is_heic:
        await message.answer("Пожалуйста, отправьте изображение (JPEG, PNG, HEIC, WebP).")
        return

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)

    filename = doc.file_name or "photo.jpg"
    await _process_and_reply(message, buf.getvalue(), filename)


# ── Приём фото (Telegram сжимает, но принимаем) ────────────────────────────────

@dp.message(F.photo)
async def recv_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)

    await message.answer(
        "💡 Для максимального качества отправляйте фото *файлом*\n"
        "(скрепка → Файл → выберите фото)",
        parse_mode="Markdown",
    )
    await _process_and_reply(message, buf.getvalue(), "photo.jpg")


# ── Run ────────────────────────────────────────────────────────────────────────

async def main():
    logger.info("Bot starting…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
