"""
broadcast.py — Retouch Lab
Система рассылок для всех пользователей бота.
Команда /broadcast — только для ADMIN_ID.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import aiosqlite
from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest

logger = logging.getLogger(__name__)

DB_PATH  = Path(__file__).parent / "retouch_lab.db"
ADMIN_ID = int(os.getenv("ADMIN_CHAT_ID", "532189427"))

# Задержка между сообщениями — чтобы не словить flood limit
# Telegram: 30 сообщений в секунду глобально, 1 сообщение/сек на пользователя
BATCH_SIZE  = 25    # сообщений за раз
BATCH_DELAY = 1.5   # секунд между батчами


# ══════════════════════════════════════════════════════════════════════════════
# Получение пользователей
# ══════════════════════════════════════════════════════════════════════════════

async def get_all_users() -> list[dict]:
    """Возвращает всех активных пользователей из БД."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT telegram_id, username, first_name
            FROM users
            WHERE telegram_id IS NOT NULL
            ORDER BY created_at ASC
        """) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def mark_user_inactive(telegram_id: int):
    """Помечает пользователя как неактивного (заблокировал бота)."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Добавляем колонку если нет
        try:
            await db.execute(
                "ALTER TABLE users ADD COLUMN is_inactive INTEGER DEFAULT 0"
            )
            await db.commit()
        except Exception:
            pass
        await db.execute(
            "UPDATE users SET is_inactive = 1 WHERE telegram_id = ?",
            (telegram_id,)
        )
        await db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Отправка одного сообщения
# ══════════════════════════════════════════════════════════════════════════════

async def send_to_user(
    bot: Bot,
    telegram_id: int,
    text: str,
) -> str:
    """
    Отправляет сообщение пользователю.
    Возвращает: 'ok' | 'blocked' | 'error'
    """
    try:
        await bot.send_message(
            chat_id=telegram_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return "ok"

    except TelegramForbiddenError:
        # Пользователь заблокировал бота
        await mark_user_inactive(telegram_id)
        return "blocked"

    except TelegramBadRequest as e:
        logger.warning("BadRequest for user %d: %s", telegram_id, e)
        return "error"

    except Exception as e:
        logger.error("Error sending to %d: %s", telegram_id, e)
        return "error"


# ══════════════════════════════════════════════════════════════════════════════
# Broadcast
# ══════════════════════════════════════════════════════════════════════════════

async def broadcast(
    bot: Bot,
    text: str,
    progress_callback=None,
) -> dict:
    """
    Отправляет сообщение всем пользователям батчами.
    Возвращает статистику: {total, sent, blocked, errors}
    """
    users = await get_all_users()
    total   = len(users)
    sent    = 0
    blocked = 0
    errors  = 0

    logger.info("Broadcast started: %d users", total)

    for i, user in enumerate(users):
        uid = user["telegram_id"]

        result = await send_to_user(bot, uid, text)

        if result == "ok":
            sent += 1
        elif result == "blocked":
            blocked += 1
        else:
            errors += 1

        # Progress callback каждые 10 пользователей
        if progress_callback and (i + 1) % 10 == 0:
            await progress_callback(i + 1, total, sent, blocked, errors)

        # Задержка между батчами
        if (i + 1) % BATCH_SIZE == 0:
            await asyncio.sleep(BATCH_DELAY)
        else:
            await asyncio.sleep(0.05)  # 50ms между сообщениями

    logger.info(
        "Broadcast done: total=%d sent=%d blocked=%d errors=%d",
        total, sent, blocked, errors
    )

    return {
        "total":   total,
        "sent":    sent,
        "blocked": blocked,
        "errors":  errors,
    }