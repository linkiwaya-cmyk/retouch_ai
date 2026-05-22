ёmodule.exports = {
  apps: [
    {
      name: "retouch-lab",
      script: "bot.py",
      interpreter: "python3",
      cwd: "/workspace/retouch_ai",

      // Даём боту 10 сек на завершение polling перед рестартом
      // Это главная причина TelegramConflictError
      kill_timeout:  10000,   // 10 сек ждём graceful shutdown
      restart_delay: 8000,    // 8 сек паузы перед новым запуском
      listen_timeout: 15000,  // 15 сек на старт

      max_restarts: 10,
      autorestart: true,
      watch: false,

      env: {
        PYTHONUNBUFFERED: "1",
      },

      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};