import asyncio
import logging
import os
import io

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
RUNPOD_URL = os.getenv("RUNPOD_URL", "https://ychsinzriqqmll-8000.proxy.runpod.net/process-image")

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
        "Я *Retouch Lab* — AI бот для профессиональной обработки фото.\n\n"
        "Что я умею:\n\n"
        "✨ выравнивать свет\n"
        "🎨 улучшать цвет\n"
        "🧴 делать natural skin tone\n"
        "📷 повышать качество изображения\n\n"
        "Выберите действие ниже 👇",
        reply_markup=main_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "📸 Обработать фото")
async def process_photo_menu(message: Message):
    await message.answer(
        "📸 *Обработка фото*\n\n"
        "Отправьте фотографию *файлом* (скрепка → документ) или просто фото.\n\n"
        "Что я сделаю:\n\n"
        "✨ уберу дефекты кожи\n"
        "🎨 сохраню натуральный тон\n"
        "🧴 natural skin tone\n"
        "📷 повышу детализацию\n\n"
        "Жду фото 📸",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "✨ Улучшить качество")
async def enhance_menu(message: Message):
    await message.answer(
        "✨ *Улучшение качества*\n\n"
        "AI восстановит:\n\n"
        "• старые фотографии\n"
        "• размытые фото\n"
        "• фото низкого качества\n\n"
        "Отправьте фотографию файлом или как фото.",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "💎 Подписка")
async def subscription(message: Message):
    await message.answer(
        "💎 *Подписка Retouch Lab*\n\n"
        "Выберите удобный тариф:\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🗓 *1 месяц*\n"
        "14$ / ~1 200 сом\n"
        "До 300 фото\n\n"
        "🗓 *3 месяца*\n"
        "40$ / ~3 500 сом\n"
        "До 1 000 фото · экономия 17%\n\n"
        "🗓 *6 месяцев*\n"
        "80$ / ~7 000 сом\n"
        "До 2 500 фото · экономия 33%\n\n"
        "🗓 *1 год*\n"
        "115$ / ~9 999 сом\n"
        "Безлимит · экономия 52%\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "💍 *Скоро:* эксклюзивная функция для свадебных фотографов — "
        "пакетная обработка галереи с единым стилем ретуши.\n\n"
        "Для оплаты и вопросов напишите нам — @linkaway\\_support",
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
        "AI автоматически:\n\n"
        "✨ убирает дефекты кожи\n"
        "🎨 сохраняет натуральный тон\n"
        "🧴 natural skin tone\n"
        "📷 повышает детализацию\n\n"
        "То, что раньше занимало 20 минут ретуши — теперь занимает несколько секунд.\n\n"
        "🔗 *Linkiway* — технологии для творческих людей.",
        reply_markup=back_menu,
        parse_mode="Markdown",
    )


@dp.message(F.text == "⬅️ Назад")
async def back(message: Message):
    await message.answer("Главное меню 👇", reply_markup=main_menu)


# ── Общая функция отправки в API ───────────────────────────────────────────────

async def _send_to_api(image_bytes: bytes, user_id: str, filename: str = "photo.jpg") -> bytes | None:
    """Отправляет изображение в FastAPI, возвращает байты результата или None."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
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
                    logger.error("API error: %s %s", resp.status, await resp.text())
                    return None
                return await resp.read()

    except asyncio.TimeoutError:
        logger.error("API timeout")
        return "timeout"
    except Exception as exc:
        logger.exception("API request failed: %s", exc)
        return None


async def _process_and_reply(message: Message, image_bytes: bytes, filename: str = "photo.jpg"):
    """Общий pipeline: скачал → отправил в API → ответил файлом."""
    user_id = str(message.from_user.id)
    status_msg = await message.answer("⏳ Обрабатываю фото, подождите...")

    result = await _send_to_api(image_bytes, user_id, filename)

    await status_msg.delete()

    if result == "limit":
        await message.answer("❌ Лимит обработок исчерпан (1000 изображений).")
        return
    if result == "timeout":
        await message.answer("❌ Сервер не ответил. Попробуйте через минуту.")
        return
    if result is None:
        await message.answer("❌ Ошибка обработки. Попробуйте ещё раз.")
        return

    # Отправляем как документ — сохраняет оригинальное качество (не сжимает)
    output_file = BufferedInputFile(result, filename="retouched.jpg")
    await message.answer_document(
        output_file,
        caption="✨ Готово! Файл сохранён в оригинальном качестве.",
    )


# ── Приём фото как ДОКУМЕНТ (несжатое, лучшее качество) ───────────────────────

@dp.message(F.document)
async def get_photo_as_document(message: Message):
    doc = message.document

    # Проверяем что это изображение
    if doc.mime_type and not doc.mime_type.startswith("image/"):
        await message.answer("Пожалуйста, отправьте изображение (JPEG, PNG).")
        return

    # Скачиваем
    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    await _process_and_reply(message, image_bytes, filename=doc.file_name or "photo.jpg")


# ── Приём фото как ФОТО (сжатое Telegram'ом, но тоже принимаем) ───────────────

@dp.message(F.photo)
async def get_photo_as_photo(message: Message):
    # Берём самое большое разрешение
    photo = message.photo[-1]

    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    image_bytes = buf.getvalue()

    await message.answer(
        "💡 Совет: для лучшего качества отправляйте фото *файлом* (скрепка → документ).",
        parse_mode="Markdown",
    )
    await _process_and_reply(message, image_bytes, filename="photo.jpg")


# ── Запуск ─────────────────────────────────────────────────────────────────────

async def main():
    logger.info("Bot started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
