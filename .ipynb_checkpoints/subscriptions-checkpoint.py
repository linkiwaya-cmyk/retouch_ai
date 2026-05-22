import aiosqlite
from datetime import datetime

DB_NAME = "retouch_lab.db"

async def check_active_subscription(telegram_id: int) -> bool:

    async with aiosqlite.connect(DB_NAME) as db:

        cursor = await db.execute(
            """
            SELECT end_date, active
            FROM subscriptions
            WHERE telegram_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (telegram_id,)
        )

        row = await cursor.fetchone()

        if not row:
            return False

        end_date, active = row

        if active != 1:
            return False

        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except:
            return False

        return end_date_obj > datetime.now()