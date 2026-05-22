module.exports = {
  apps: [{
    name: "retouch-lab",
    script: "bot.py",
    interpreter: "python3",
    cwd: "/workspace/retouch_ai",
    kill_timeout: 10000,
    restart_delay: 8000,
    max_restarts: 10,
    autorestart: true,
    watch: false,
    env: { PYTHONUNBUFFERED: "1" },
    log_date_format: "YYYY-MM-DD HH:mm:ss"
  }]
};
