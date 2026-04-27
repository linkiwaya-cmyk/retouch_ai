"""
bot.py v6

Изменения:
- answer_document без caption → отдельное сообщение "Готово ✨"
- Убран любой лишний текст под файлом
- HEIC определяется по extension И mime type
- sendDocument только
"""
import asyncio, io, logging, os
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
        "Поддерживаю: JPEG, PNG, HEIC \\(iPhone\\), WebP",
        reply_markup=main_menu,
        parse_mode="MarkdownV2",
    )


@dp.message(F.text == "📸 Обработать фото")
async def menu_process(message: Message):
    await message.answer(
        "📸 Отправьте фото *файлом* для максимального качества\n"
        "или просто как фотографию.\n\n"
        "iPhone \\(HEIC\\), Android \\(JPEG/PNG\\) — всё принимается 👇",
        reply_markup=back_menu,
        parse_mode="MarkdownV2",
    )


@dp.message(F.text == "✨ Улучшить качество")
async def menu_enhance(message: Message):
    await message.answer(
        "✨ Отправьте фото — я улучшу:\n\n"
        "• чистоту кожи\n"
        "• выровняю тон\n"
        "• аккуратный Dodge & Burn",
        reply_markup=back_menu,
    )


@dp.message(F.text == "💎 Подписка")
async def menu_sub(message: Message):
    await message.answer(
        "💎 *Подписка Retouch Lab*\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📅 *1 месяц* — 20\\$ / \\~1 800 сом · до 300 фото\n\n"
        "📅 *3 месяца* — 50\\$ / \\~4 500 сом · экономия 17%\n\n"
        "📅 *6 месяцев* — 80\\$ / \\~7 000 сом · экономия 33%\n\n"
        "📅 *1 год* — 115\\$ / \\~10 000 сом · экономия 52% 🔥\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "✅ AI ретушь кожи\n"
        "✅ Оригинальное качество\n"
        "✅ Natural skin tone\n\n"
        "💍 *Скоро* — функция для свадебных фотографов\\!\n\n"
        "По вопросам: @linkiway\\_support",
        reply_markup=back_menu,
        parse_mode="MarkdownV2",
    )


@dp.message(F.text == "ℹ️ О боте")
async def menu_about(message: Message):
    await message.answer(
        "ℹ️ *Retouch Lab* — AI ретушёр от *Linkiway*\n\n"
        "📸 фотографам · 🎥 блогерам · 💍 свадебным фотографам\n\n"
        "То, что раньше занимало 20 минут — теперь секунды\\.\n\n"
        "🔗 *Linkiway* — технологии для творческих людей\\.",
        reply_markup=back_menu,
        parse_mode="MarkdownV2",
    )


@dp.message(F.text == "⬅️ Назад")
async def back(message: Message):
    await message.answer("Главное меню 👇", reply_markup=main_menu)


# ── API ────────────────────────────────────────────────────────────────────────

async def _send_to_api(data: bytes, user_id: str, filename: str) -> bytes | str:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
            form = aiohttp.FormData()
            form.add_field("file", data, filename=filename, content_type="application/octet-stream")
            form.add_field("user_id", user_id)
            async with s.post(RUNPOD_URL, data=form) as r:
                if r.status == 429: return "limit"
                if r.status != 200:
                    logger.error("API %d: %s", r.status, await r.text())
                    return "error"
                return await r.read()
    except asyncio.TimeoutError:
        return "timeout"
    except Exception as e:
        logger.exception("API error: %s", e)
        return "error"


async def _process_and_reply(message: Message, data: bytes, filename: str):
    uid = str(message.from_user.id)
    logger.info("Recv %.2f MB from %s file=%s", len(data)/1024/1024, uid, filename)

    status = await message.answer("⏳ Обрабатываю…")
    result = await _send_to_api(data, uid, filename)
    await status.delete()

    if result == "limit":
        await message.answer("❌ Лимит обработок исчерпан.")
        return
    if result == "timeout":
        await message.answer("❌ Сервер не ответил. Попробуйте позже.")
        return
    if not isinstance(result, bytes):
        await message.answer("❌ Ошибка обработки. Попробуйте ещё раз.")
        return

    stem     = filename.rsplit(".", 1)[0]
    out_name = f"retouched_{stem}.jpg"

    # Отправляем файл БЕЗ caption
    await message.answer_document(
        BufferedInputFile(result, filename=out_name),
        caption=None,   # явно None — никакого текста под файлом
    )
    # Отдельное сообщение "Готово ✨" — чисто, без filename
    await message.answer("Готово ✨")
    logger.info("Sent %.2f MB to %s", len(result)/1024/1024, uid)


# ── Приём документов (HEIC, JPEG, PNG, WebP) ───────────────────────────────────

_HEIC_EXTS  = {".heic", ".heif"}
_HEIC_MIMES = {"image/heic", "image/heif", "image/heif-sequence"}
_IMG_MIMES  = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@dp.message(F.document)
async def recv_document(message: Message):
    doc  = message.document
    mime = (doc.mime_type or "").lower()
    name = doc.file_name or "photo.jpg"
    ext  = ("." + name.rsplit(".", 1)[-1]).lower() if "." in name else ""

    is_heic  = ext in _HEIC_EXTS or mime in _HEIC_MIMES
    is_image = mime in _IMG_MIMES or mime.startswith("image/") or is_heic

    if not is_image:
        await message.answer("Пожалуйста, отправьте изображение \\(JPEG, PNG, HEIC, WebP\\)\\.",
                             parse_mode="MarkdownV2")
        return

    file = await bot.get_file(doc.file_id)
    buf  = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    await _process_and_reply(message, buf.getvalue(), name)


# ── Приём сжатых фото ──────────────────────────────────────────────────────────

@dp.message(F.photo)
async def recv_photo(message: Message):
    photo = message.photo[-1]
    file  = await bot.get_file(photo.file_id)
    buf   = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)

    await message.answer(
        "💡 Для максимального качества отправляйте фото *файлом*\n"
        "\\(скрепка → Файл → выберите фото\\)",
        parse_mode="MarkdownV2",
    )
    await _process_and_reply(message, buf.getvalue(), "photo.jpg")


# ── Run ────────────────────────────────────────────────────────────────────────

async def main():
    logger.info("Bot v6 starting…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
