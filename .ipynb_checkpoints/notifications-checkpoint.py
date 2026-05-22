"""
notifications.py — Retouch Lab
Ежедневные уведомления об окончании подписки: за 3 дня, за 1 день, в день окончания.
Запускается через APScheduler внутри bot.py.
"""

import logging
from aiogram import Bot
from database import get_expiring_subscriptions

logger = logging.getLogger(__name__)


async def send_expiry_notifications(bot: Bot):
    """
    Проверяет подписки и отправляет уведомления.
    Вызывается ежедневно в 10:00.
    """
    # За 3 дня
    for user in await get_expiring_subscriptions(days=3):
        try:
            await bot.send_message(
                chat_id=user["telegram_id"],
                text=(
                    "⚠️ <b>Подписка заканчивается через 3 дня</b>\n\n"
                    "Ваша подписка Retouch Lab истекает <b>"
                    + user["end_date"][:10].replace("-", ".").split(".")[::-1].__class__(
                        reversed(user["end_date"][:10].split("-"))
                    ).__class__(".".join(reversed(user["end_date"][:10].split("-"))))
                    + "</b>\n\n"
                    "Продлите подписку чтобы не прерывать ретушь 💎"
                ),
                parse_mode="HTML",
            )
            logger.info("Notified (3 days): user=%d", user["telegram_id"])
        except Exception as e:
            logger.error("Notify error user=%d: %s", user["telegram_id"], e)

    # За 1 день
    for user in await get_expiring_subscriptions(days=1):
        try:
            end_pretty = ".".join(reversed(user["end_date"][:10].split("-")))
            await bot.send_message(
                chat_id=user["telegram_id"],
                text=(
                    "🔔 <b>Подписка заканчивается завтра!</b>\n\n"
                    f"Retouch Lab истекает <b>{end_pretty}</b>\n\n"
                    "Продлите сегодня — нажмите 💎 Подписка"
                ),
                parse_mode="HTML",
            )
            logger.info("Notified (1 day): user=%d", user["telegram_id"])
        except Exception as e:
            logger.error("Notify error user=%d: %s", user["telegram_id"], e)

    # В день окончания
    for user in await get_expiring_subscriptions(days=0):
        try:
            await bot.send_message(
                chat_id=user["telegram_id"],
                text=(
                    "❌ <b>Подписка Retouch Lab закончилась сегодня</b>\n\n"
                    "Обработка фото приостановлена.\n\n"
                    "Оформите новую подписку — нажмите 💎 Подписка"
                ),
                parse_mode="HTML",
            )
            logger.info("Notified (expired): user=%d", user["telegram_id"])
        except Exception as e:
            logger.error("Notify error user=%d: %s", user["telegram_id"], e)