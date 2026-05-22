"""
reminders.py — Retouch Lab
Retention система: напоминания + акция для неактивных пользователей.
Запускается через APScheduler каждый день.
"""

import logging
import random
from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError
from database import get_inactive_users, mark_reminder_sent, mark_user_inactive

logger = logging.getLogger(__name__)

# ── Тексты напоминаний (rotation) ────────────────────────────────────────────
# Стиль: мягко, эстетично, без агрессии

REMINDER_MESSAGES = [
    (
        "✨ Привет!\n\n"
        "Твои фото всё ещё ждут красивую обработку.\n\n"
        "Попробуй отправить ещё один кадр — иногда одна аккуратная ретушь "
        "меняет всё настроение снимка 📸\n\n"
        "💎 Оформи подписку и обрабатывай без ограничений."
    ),
    (
        "📸 Как твои фото?\n\n"
        "Retouch Lab готов к работе — natural AI-ретушь, "
        "оригинальное разрешение, без пластика.\n\n"
        "🎉 <b>Акция этой недели:</b> подписка на месяц — всего <b>799 сом</b>\n\n"
        "💎 Нажми 'Подписка' чтобы продолжить."
    ),
    (
        "👋 Привет!\n\n"
        "Помнишь как выглядело твоё фото после обработки? ✨\n\n"
        "Retouch Lab сохраняет родинки, текстуру и натуральность — "
        "просто делает кожу чище.\n\n"
        "💎 Подписка открывает неограниченную обработку."
    ),
]

# ── Акционное сообщение ───────────────────────────────────────────────────────

PROMO_MESSAGE = (
    "🎉 <b>Специальное предложение!</b>\n\n"
    "До конца недели подписка на <b>1 месяц — 799 сом</b> вместо 990 сом\n\n"
    "✦ Natural AI-ретушь\n"
    "✦ Оригинальное разрешение 4K / 24MP\n"
    "✦ Без потери качества\n\n"
    "Нажми 💎 Подписка чтобы воспользоваться акцией 👇"
)

# ── Сообщение о возвращении бота ─────────────────────────────────────────────

RELAUNCH_MESSAGE = (
    "✦ <b>Retouch Lab снова онлайн</b>\n\n"
    "Мы провели обновления:\n"
    "• Улучшена стабильность работы\n"
    "• Сохранение оригинального разрешения 4K / 24MP\n"
    "• Более natural результат ретуши\n"
    "• Ускорена обработка\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "🎉 <b>Акция в честь обновления:</b>\n"
    "Подписка на 1 месяц — <b>799 сом</b> вместо 990 сом\n\n"
    "Нажми 💎 Подписка чтобы начать 👇"
)


# ══════════════════════════════════════════════════════════════════════════════
# Отправка напоминаний
# ══════════════════════════════════════════════════════════════════════════════

async def send_reminders(bot: Bot):
    """
    Ежедневная проверка неактивных пользователей.
    Отправляет мягкое напоминание тем кто:
    - пробовал trial
    - не купил подписку
    - не активен 3+ дней
    - ещё не получал reminder
    """
    users = await get_inactive_users(days=3)
    logger.info("Reminders: found %d inactive users", len(users))

    sent = 0
    for user in users:
        uid = user["telegram_id"]
        text = random.choice(REMINDER_MESSAGES)

        try:
            await bot.send_message(
                chat_id=uid,
                text=text,
                parse_mode="HTML",
            )
            await mark_reminder_sent(uid)
            sent += 1
            logger.info("Reminder sent: user=%d", uid)

        except TelegramForbiddenError:
            await mark_user_inactive(uid)
            logger.info("User blocked bot: %d", uid)

        except Exception as e:
            logger.error("Reminder error user=%d: %s", uid, e)

    logger.info("Reminders done: sent=%d/%d", sent, len(users))


async def send_promo(bot: Bot, user_ids: list[int] = None):
    """
    Отправляет акционное сообщение.
    Если user_ids не указан — отправляет всем неактивным.
    """
    from database import get_all_users
    from broadcast import send_to_user

    if user_ids is None:
        users = await get_inactive_users(days=1)
        user_ids = [u["telegram_id"] for u in users]

    logger.info("Promo: sending to %d users", len(user_ids))
    sent = 0
    for uid in user_ids:
        result = await send_to_user(bot, uid, PROMO_MESSAGE)
        if result == "ok":
            sent += 1
    logger.info("Promo done: sent=%d", sent)