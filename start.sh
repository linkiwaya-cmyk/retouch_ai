#!/bin/bash

cd /workspace/retouch_ai

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing aiogram..."
pip install aiogram apscheduler

echo "Installing PM2..."
npm install -g pm2

echo "Stopping old PM2 process..."
pm2 delete all || true

echo "Starting bot..."
pm2 start ecosystem.config.js

echo "Saving PM2..."
pm2 save

echo "Done."
