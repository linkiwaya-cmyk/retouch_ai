"""
database.py — Retouch Lab
SQLite: users, subscriptions, payments
Самоинициализирующийся — таблицы создаются автоматически при первом запросе.
"""

import aiosqlite
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "retouch_lab.db"

TRIAL_LIMIT = 3  # максимум бесплатных обработок

PLAN_DAYS   = {"1m": 30,  "3m": 90,  "6m": 180, "1y": 365}
PLAN_PRICES = {"1m": 990, "3m": 2490,"6m": 4990,"1y": 8990}
PLAN_NAMES  = {"1m": "1 месяц","3m": "3 месяца","6m": "6 месяцев","1y": "1 год"}


# ══════════════════════════════════════════════════════════════════════════════
# Внутренняя функция — гарантирует что все таблицы существуют
# ══════════════════════════════════════════════════════════════════════════════

async def _ensure_tables(db):
    """Создаёт все таблицы и колонки если их нет. Вызывается при каждом запросе."""
    await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            telegram_id        INTEGER PRIMARY KEY,
            username           TEXT,
            first_name         TEXT,
            created_at         TEXT DEFAULT (datetime('now')),
            trial_photos_count INTEGER DEFAULT 0
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL,
            plan_type   TEXT NOT NULL,
            start_date  TEXT NOT NULL,
            end_date    TEXT NOT NULL,
            is_active   INTEGER DEFAULT 1
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id        INTEGER NOT NULL,
            amount             INTEGER NOT NULL,
            plan_type          TEXT NOT NULL,
            payment_status     TEXT DEFAULT 'pending',
            created_at         TEXT DEFAULT (datetime('now')),
            screenshot_file_id TEXT
        )
    """)
    # Миграции — добавляем колонки если их нет в старой БД
    for sql in [
        "ALTER TABLE users ADD COLUMN username TEXT",
        "ALTER TABLE users ADD COLUMN first_name TEXT",
        "ALTER TABLE users ADD COLUMN trial_photos_count INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN last_active TEXT",
        "ALTER TABLE users ADD COLUMN reminder_sent INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN is_inactive INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN language TEXT DEFAULT 'ru'",
    ]:
        try:
            await db.execute(sql)
        except Exception:
            pass
    # Таблица аналитики
    await db.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL,
            event       TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    await db.commit()


async def init_db():
    """Публичная инициализация — вызывать при старте бота."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
    logger.info("Database ready: %s", DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════════════════

async def add_user(telegram_id: int, username: str | None, first_name: str | None):
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await db.execute("""
            INSERT OR IGNORE INTO users (telegram_id, username, first_name, trial_photos_count, last_active)
            VALUES (?, ?, ?, 0, ?)
        """, (telegram_id, username, first_name, now))
        # Обновляем last_active при каждом визите
        await db.execute("""
            UPDATE users SET last_active = ?, is_inactive = 0
            WHERE telegram_id = ?
        """, (now, telegram_id))
        await db.commit()


async def get_user(telegram_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
# Trial — счётчик бесплатных обработок
# ══════════════════════════════════════════════════════════════════════════════

async def get_trial_count(telegram_id: int) -> int:
    """Возвращает сколько фото пользователь обработал бесплатно."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        async with db.execute(
            "SELECT trial_photos_count FROM users WHERE telegram_id = ?",
            (telegram_id,)
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return 0
            return int(row[0]) if row[0] is not None else 0


async def increment_trial_count(telegram_id: int) -> int:
    """
    Увеличивает счётчик на 1 после УСПЕШНОЙ обработки.
    Возвращает новое значение.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        # Убеждаемся что пользователь есть
        await db.execute("""
            INSERT OR IGNORE INTO users (telegram_id, trial_photos_count)
            VALUES (?, 0)
        """, (telegram_id,))
        # Увеличиваем счётчик
        await db.execute("""
            UPDATE users
            SET trial_photos_count = trial_photos_count + 1
            WHERE telegram_id = ?
        """, (telegram_id,))
        await db.commit()
        # Читаем новое значение
        async with db.execute(
            "SELECT trial_photos_count FROM users WHERE telegram_id = ?",
            (telegram_id,)
        ) as cur:
            row = await cur.fetchone()
            count = int(row[0]) if row else 0
    logger.info("Trial count incremented: user=%d → count=%d/%d", telegram_id, count, TRIAL_LIMIT)
    return count


async def get_free_trial_used(telegram_id: int) -> bool:
    count = await get_trial_count(telegram_id)
    return count >= TRIAL_LIMIT


async def mark_free_trial_used(telegram_id: int):
    await increment_trial_count(telegram_id)


# ══════════════════════════════════════════════════════════════════════════════
# Subscriptions
# ══════════════════════════════════════════════════════════════════════════════

async def check_active_subscription(telegram_id: int) -> dict | None:
    """
    Проверяет активную подписку.
    Использует SQLite datetime('now') для сравнения — без timezone проблем.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM subscriptions
            WHERE telegram_id = ?
              AND is_active = 1
              AND datetime(end_date) > datetime('now')
            ORDER BY end_date DESC LIMIT 1
        """, (telegram_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def activate_subscription(telegram_id: int, plan_type: str) -> str:
    days       = PLAN_DAYS.get(plan_type, 30)
    # Используем UTC чтобы избежать timezone проблем после перезапуска
    from datetime import timezone
    start      = datetime.now(timezone.utc).replace(tzinfo=None)
    end        = start + timedelta(days=days)
    start_str  = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str    = end.strftime("%Y-%m-%d %H:%M:%S")
    end_pretty = end.strftime("%d.%m.%Y")

    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute("""
            UPDATE subscriptions SET is_active = 0
            WHERE telegram_id = ? AND is_active = 1
        """, (telegram_id,))
        await db.execute("""
            INSERT INTO subscriptions (telegram_id, plan_type, start_date, end_date, is_active)
            VALUES (?, ?, ?, ?, 1)
        """, (telegram_id, plan_type, start_str, end_str))
        await db.commit()

    logger.info("Subscription activated: user=%d plan=%s until=%s", telegram_id, plan_type, end_pretty)
    return end_pretty


async def deactivate_subscription(telegram_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute("""
            UPDATE subscriptions SET is_active = 0
            WHERE telegram_id = ? AND is_active = 1
        """, (telegram_id,))
        await db.commit()


async def get_expiring_subscriptions(days: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row
        # Используем SQLite date arithmetic вместо Python datetime
        async with db.execute("""
            SELECT s.telegram_id, s.end_date, u.first_name
            FROM subscriptions s
            JOIN users u ON s.telegram_id = u.telegram_id
            WHERE s.is_active = 1
              AND date(s.end_date) = date('now', ? || ' days')
        """, (f"+{days}",)) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Payments
# ══════════════════════════════════════════════════════════════════════════════

async def create_payment(telegram_id: int, plan_type: str, screenshot_file_id: str) -> int:
    amount = PLAN_PRICES.get(plan_type, 0)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        async with db.execute("""
            INSERT INTO payments (telegram_id, amount, plan_type, payment_status, screenshot_file_id)
            VALUES (?, ?, ?, 'pending', ?)
        """, (telegram_id, amount, plan_type, screenshot_file_id)) as cur:
            payment_id = cur.lastrowid
        await db.commit()
    logger.info("Payment created: id=%d user=%d plan=%s", payment_id, telegram_id, plan_type)
    return payment_id


async def get_payment(payment_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM payments WHERE id = ?", (payment_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def update_payment_status(payment_id: int, status: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute(
            "UPDATE payments SET payment_status = ? WHERE id = ?",
            (status, payment_id)
        )
        await db.commit()


async def log_analytics_event(telegram_id: int, event: str):
    """Логирует аналитическое событие."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute(
            "INSERT INTO events (telegram_id, event) VALUES (?, ?)",
            (telegram_id, event)
        )
        await db.commit()


async def get_analytics_summary() -> dict:
    """Возвращает сводку аналитики за последние 7 дней."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        result = {}
        for event in [
            "photo_processed", "paywall_shown",
            "photo_processing_started", "photo_processing_error"
        ]:
            async with db.execute("""
                SELECT COUNT(*) as cnt FROM events
                WHERE event = ?
                AND datetime(created_at) > datetime('now', '-7 days')
            """, (event,)) as cur:
                row = await cur.fetchone()
                result[event] = row["cnt"] if row else 0
        # Конверсия
        shown = result.get("paywall_shown", 0)
        async with db.execute("""
            SELECT COUNT(*) as cnt FROM payments
            WHERE payment_status = 'approved'
            AND datetime(created_at) > datetime('now', '-7 days')
        """) as cur:
            row = await cur.fetchone()
            result["subscriptions_bought"] = row["cnt"] if row else 0
        result["conversion_pct"] = (
            round(result["subscriptions_bought"] / shown * 100, 1)
            if shown > 0 else 0
        )
        return result


async def get_user_language(telegram_id: int) -> str:
    """Возвращает язык пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        async with db.execute(
            "SELECT language FROM users WHERE telegram_id = ?",
            (telegram_id,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row and row[0] else "ru"


async def set_user_language(telegram_id: int, language: str):
    """Устанавливает язык пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute(
            "UPDATE users SET language = ? WHERE telegram_id = ?",
            (language, telegram_id)
        )
        await db.commit()


async def update_last_active(telegram_id: int):
    """Обновляет время последней активности пользователя."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute(
            "UPDATE users SET last_active = datetime('now'), is_inactive = 0 WHERE telegram_id = ?",
            (telegram_id,)
        )
        await db.commit()


async def get_inactive_users(days: int = 3) -> list[dict]:
    """
    Возвращает пользователей которые:
    - использовали trial (trial_photos_count > 0)
    - не имеют активной подписки
    - неактивны N дней
    - ещё не получали reminder
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT u.telegram_id, u.first_name, u.username,
                   u.trial_photos_count, u.last_active, u.reminder_sent
            FROM users u
            WHERE u.trial_photos_count > 0
              AND u.reminder_sent = 0
              AND u.is_inactive = 0
              AND (
                  u.last_active IS NULL
                  OR datetime(u.last_active) < datetime('now', ? || ' days')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM subscriptions s
                  WHERE s.telegram_id = u.telegram_id
                    AND s.is_active = 1
                    AND s.end_date > datetime('now')
              )
        """, (f"-{days}",)) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def mark_reminder_sent(telegram_id: int):
    """Помечает что reminder отправлен — не спамить."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute(
            "UPDATE users SET reminder_sent = 1 WHERE telegram_id = ?",
            (telegram_id,)
        )
        await db.commit()


async def reset_reminder_flag(telegram_id: int):
    """Сбрасывает флаг reminder (например при новой активности)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        await db.execute(
            "UPDATE users SET reminder_sent = 0 WHERE telegram_id = ?",
            (telegram_id,)
        )
        await db.commit()


async def get_pending_payments() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_tables(db)
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT p.*, u.username, u.first_name
            FROM payments p
            JOIN users u ON p.telegram_id = u.telegram_id
            WHERE p.payment_status = 'pending'
            ORDER BY p.created_at ASC
        """) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
