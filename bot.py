"""
Telegram bot — photo retouching integration.

Only the essential parts:
  - receive photo
  - send to backend API
  - return result via sendDocument (not sendPhoto)
  - per-user usage enforced on backend side
"""

from __future__ import annotations

import io
import logging
import os

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
API_URL = os.environ.get("RETOUCH_API_URL", "http://localhost:8000")

# ── handlers ───────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Отправьте фото для профессиональной ретуши.\n"
        "Я сохраню текстуру кожи и уберу дефекты.",
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    msg = update.message

    # ── 1. Download the highest-resolution version ─────────────────────────
    photo = msg.photo[-1]  # last element = largest
    file = await context.bot.get_file(photo.file_id)

    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)
    image_bytes = buf.read()

    status_msg = await msg.reply_text("⏳ Обрабатываю…")

    # ── 2. Send to backend ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_URL}/process-image",
                files={"file": ("photo.jpg", image_bytes, "image/jpeg")},
                data={"user_id": user_id},
            )

        if response.status_code == 429:
            await status_msg.edit_text("❌ Лимит обработок исчерпан (1000 изображений).")
            return
        if response.status_code != 200:
            await status_msg.edit_text(f"❌ Ошибка сервера: {response.status_code}")
            return

        result_bytes = response.content

    except httpx.TimeoutException:
        await status_msg.edit_text("❌ Сервер не ответил вовремя. Попробуйте позже.")
        return
    except Exception as exc:
        logger.exception("API request failed")
        await status_msg.edit_text(f"❌ Ошибка: {exc}")
        return

    # ── 3. Send result as document (sendDocument, NOT sendPhoto) ───────────
    await context.bot.send_document(
        chat_id=msg.chat_id,
        document=io.BytesIO(result_bytes),
        filename="retouched.jpg",
        caption="✅ Готово! Файл сохраняет оригинальное качество.",
        reply_to_message_id=msg.message_id,
    )
    await status_msg.delete()


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Accept photos sent as files (uncompressed)."""
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("Пожалуйста, отправьте изображение.")
        return

    user_id = str(update.effective_user.id)
    file = await context.bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)
    image_bytes = buf.read()

    status_msg = await update.message.reply_text("⏳ Обрабатываю файл…")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_URL}/process-image",
                files={"file": (doc.file_name or "image.jpg", image_bytes, doc.mime_type)},
                data={"user_id": user_id},
            )

        if response.status_code == 429:
            await status_msg.edit_text("❌ Лимит обработок исчерпан.")
            return
        if response.status_code != 200:
            await status_msg.edit_text(f"❌ Ошибка сервера: {response.status_code}")
            return

        result_bytes = response.content

    except Exception as exc:
        logger.exception("API request failed")
        await status_msg.edit_text(f"❌ Ошибка: {exc}")
        return

    await context.bot.send_document(
        chat_id=update.message.chat_id,
        document=io.BytesIO(result_bytes),
        filename="retouched.jpg",
        caption="✅ Готово!",
        reply_to_message_id=update.message.message_id,
    )
    await status_msg.delete()


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    logger.info("Bot started.")
    app.run_polling()


if __name__ == "__main__":
    main()
