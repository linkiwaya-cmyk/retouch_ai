"""
backup.py — Retouch Lab
Автоматический бэкап retouch_lab.db
Запуск: python backup.py
Или через cron каждый день в 3:00:
  0 3 * * * python /workspace/retouch_ai/backup.py
"""

import shutil
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH     = Path("/workspace/retouch_ai/retouch_lab.db")
BACKUP_DIR  = Path("/workspace/retouch_ai/backups")
MAX_BACKUPS = 7  # хранить последние 7 копий


def backup():
    if not DB_PATH.exists():
        logger.error("DB not found: %s", DB_PATH)
        return

    BACKUP_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"retouch_lab_{timestamp}.db"

    shutil.copy2(DB_PATH, dest)
    logger.info("Backup created: %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)

    # Удаляем старые — оставляем только MAX_BACKUPS штук
    backups = sorted(BACKUP_DIR.glob("retouch_lab_*.db"))
    if len(backups) > MAX_BACKUPS:
        for old in backups[:-MAX_BACKUPS]:
            old.unlink()
            logger.info("Old backup removed: %s", old.name)

    logger.info("Total backups: %d", len(list(BACKUP_DIR.glob("*.db"))))


if __name__ == "__main__":
    backup()