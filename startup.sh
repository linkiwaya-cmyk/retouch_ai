#!/bin/bash
cd /workspace/retouch_ai
pip install -q aiogram apscheduler aiosqlite python-dotenv pillow pillow-heif requests numpy
if ! command -v pm2 &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1
    apt-get install -y nodejs >/dev/null 2>&1
    npm install -g pm2 >/dev/null 2>&1
fi
pm2 delete retouch-lab 2>/dev/null || true
sleep 3
pm2 start ecosystem.config.js
pm2 save
pm2 status
