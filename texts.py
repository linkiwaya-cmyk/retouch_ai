"""
texts.py — Retouch Lab
Все тексты бота на 4 языках.
RU = русский, EN = английский, VI = вьетнамский, KY = кыргызский
"""

LANGUAGES = {
    "ru": "🇷🇺 Русский",
    "en": "🇬🇧 English",
    "vi": "🇻🇳 Tiếng Việt",
    "ky": "🇰🇬 Кыргызча",
}

TEXTS = {

    # ══════════════════════════════════════════════════════════════════════════
    # ПРИВЕТСТВИЕ / WELCOME
    # ══════════════════════════════════════════════════════════════════════════

    "welcome": {
        "ru": (
            "✦ ═══════════════════ ✦\n"
            "✨ <b>R E T O U C H   L A B</b>\n"
            "<i>AI-ретушь фото прямо в Telegram</i>\n"
            "✦ ═══════════════════ ✦\n\n"
            "Профессиональная Dodge & Burn ретушь за секунды.\n"
            "Без потери качества. Без plastic-эффекта.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 режимов обработки:</b>\n\n"
            "✨ Чистая кожа — минимально, натурально\n"
            "🌿 Натуральная ретушь — для каждого дня\n"
            "💫 Объём и свет — глубина, светотень\n"
            "💄 Beauty Pro — для соцсетей и Instagram\n"
            "🌟 Журнальный стиль — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📷 4K / 24MP / HEIC / WebP\n"
            "🎁 <b>Первые 3 фото — бесплатно</b>\n\n"
            "📌 Отправляйте фото <b>файлом</b> (📎 → Файл)"
        ),
        "en": (
            "✦ ═══════════════════ ✦\n"
            "✨ <b>R E T O U C H   L A B</b>\n"
            "<i>AI photo retouching in Telegram</i>\n"
            "✦ ═══════════════════ ✦\n\n"
            "Professional Dodge & Burn retouching in seconds.\n"
            "No quality loss. No plastic effect.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 processing modes:</b>\n\n"
            "✨ Clean Skin — minimal, natural\n"
            "🌿 Natural Retouch — for everyday use\n"
            "💫 Depth & Light — volume, shadows\n"
            "💄 Beauty Pro — for social media\n"
            "🌟 Magazine Style — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📷 4K / 24MP / HEIC / WebP\n"
            "🎁 <b>First 3 photos — free</b>\n\n"
            "📌 Send photos as <b>file</b> (📎 → File)"
        ),
        "vi": (
            "✦ ═══════════════════ ✦\n"
            "✨ <b>R E T O U C H   L A B</b>\n"
            "<i>Chỉnh sửa ảnh AI ngay trong Telegram</i>\n"
            "✦ ═══════════════════ ✦\n\n"
            "Chỉnh sửa Dodge & Burn chuyên nghiệp trong vài giây.\n"
            "Không mất chất lượng. Không hiệu ứng giả tạo.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 chế độ xử lý:</b>\n\n"
            "✨ Da Sạch — tối giản, tự nhiên\n"
            "🌿 Retouch Tự Nhiên — dùng hàng ngày\n"
            "💫 Chiều Sâu & Ánh Sáng — khối lượng, bóng\n"
            "💄 Beauty Pro — cho mạng xã hội\n"
            "🌟 Phong Cách Tạp Chí — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📷 4K / 24MP / HEIC / WebP\n"
            "🎁 <b>3 ảnh đầu tiên — miễn phí</b>\n\n"
            "📌 Gửi ảnh dưới dạng <b>tệp</b> (📎 → Tệp)"
        ),
        "ky": (
            "✦ ═══════════════════ ✦\n"
            "✨ <b>R E T O U C H   L A B</b>\n"
            "<i>Telegramда AI сүрөт ретуши</i>\n"
            "✦ ═══════════════════ ✦\n\n"
            "Кесипкөй Dodge & Burn ретуши секундтарда.\n"
            "Сапат жоготпой. Пластик эффектсиз.\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 иштетүү режими:</b>\n\n"
            "✨ Таза тери — минималдуу, табигый\n"
            "🌿 Табигый ретушь — күн сайын\n"
            "💫 Көлөм жана жарык — тереңдик, жарык\n"
            "💄 Beauty Pro — социалдык тармактар үчүн\n"
            "🌟 Журнал стили — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📷 4K / 24MP / HEIC / WebP\n"
            "🎁 <b>Алгачкы 3 сүрөт — акысыз</b>\n\n"
            "📌 Сүрөттү <b>файл</b> катары жибериңиз (📎 → Файл)"
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # КНОПКИ ГЛАВНОГО МЕНЮ
    # ══════════════════════════════════════════════════════════════════════════

    "btn_try_free": {
        "ru": "🎁 Попробовать бесплатно (осталось {n} из 3)",
        "en": "🎁 Try free ({n} of 3 left)",
        "vi": "🎁 Dùng thử miễn phí (còn {n}/3)",
        "ky": "🎁 Акысыз байкап көр ({n}/3 калды)",
    },
    "btn_process": {
        "ru": "✨ Обработать фото",
        "en": "✨ Process photo",
        "vi": "✨ Xử lý ảnh",
        "ky": "✨ Сүрөттү иштет",
    },
    "btn_examples": {
        "ru": "🎥 Примеры до / после",
        "en": "🎥 Before / After examples",
        "vi": "🎥 Ví dụ trước / sau",
        "ky": "🎥 Мисалдар чейин / кийин",
    },
    "btn_subscription": {
        "ru": "💎 Подписка",
        "en": "💎 Subscription",
        "vi": "💎 Đăng ký",
        "ky": "💎 Жазылуу",
    },
    "btn_about": {
        "ru": "ℹ️ О боте",
        "en": "ℹ️ About",
        "vi": "ℹ️ Giới thiệu",
        "ky": "ℹ️ Бот жөнүндө",
    },
    "btn_support": {
        "ru": "💬 Поддержка",
        "en": "💬 Support",
        "vi": "💬 Hỗ trợ",
        "ky": "💬 Колдоо",
    },
    "btn_buy_sub": {
        "ru": "💎 Купить подписку",
        "en": "💎 Buy subscription",
        "vi": "💎 Mua đăng ký",
        "ky": "💎 Жазылуу сатып ал",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # РЕЖИМЫ ОБРАБОТКИ
    # ══════════════════════════════════════════════════════════════════════════

    "btn_mode_clean": {
        "ru": "✨ Чистая кожа",
        "en": "✨ Clean Skin",
        "vi": "✨ Da Sạch",
        "ky": "✨ Таза тери",
    },
    "btn_mode_natural": {
        "ru": "🌿 Натуральная ретушь",
        "en": "🌿 Natural Retouch",
        "vi": "🌿 Retouch Tự Nhiên",
        "ky": "🌿 Табигый ретушь",
    },
    "btn_mode_depth": {
        "ru": "💫 Объём и свет",
        "en": "💫 Depth & Light",
        "vi": "💫 Chiều Sâu & Ánh Sáng",
        "ky": "💫 Көлөм жана жарык",
    },
    "btn_mode_beauty": {
        "ru": "💄 Beauty Pro",
        "en": "💄 Beauty Pro",
        "vi": "💄 Beauty Pro",
        "ky": "💄 Beauty Pro",
    },
    "btn_mode_magazine": {
        "ru": "🌟 Журнальный стиль",
        "en": "🌟 Magazine Style",
        "vi": "🌟 Phong Cách Tạp Chí",
        "ky": "🌟 Журнал стили",
    },
    "btn_about_modes": {
        "ru": "📖 О режимах",
        "en": "📖 About modes",
        "vi": "📖 Về các chế độ",
        "ky": "📖 Режимдер жөнүндө",
    },
    "btn_back_main": {
        "ru": "⬅️ Главное меню",
        "en": "⬅️ Main menu",
        "vi": "⬅️ Menu chính",
        "ky": "⬅️ Башкы меню",
    },
    "btn_back_modes": {
        "ru": "⬅️ Назад к режимам",
        "en": "⬅️ Back to modes",
        "vi": "⬅️ Quay lại chế độ",
        "ky": "⬅️ Режимдерге кайт",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # СООБЩЕНИЯ ОБРАБОТКИ
    # ══════════════════════════════════════════════════════════════════════════

    "processing": {
        "ru": "⏳ Обрабатываю фото...",
        "en": "⏳ Processing photo...",
        "vi": "⏳ Đang xử lý ảnh...",
        "ky": "⏳ Сүрөт иштелип жатат...",
    },
    "done": {
        "ru": "✅ <b>Готово!</b> Время обработки: <b>{t} сек</b>",
        "en": "✅ <b>Done!</b> Processing time: <b>{t} sec</b>",
        "vi": "✅ <b>Xong!</b> Thời gian xử lý: <b>{t} giây</b>",
        "ky": "✅ <b>Даяр!</b> Иштетүү убактысы: <b>{t} сек</b>",
    },
    "done_details": {
        "ru": (
            "📐 Разрешение: оригинал сохранён\n"
            "🔍 Текстура кожи: сохранена\n"
            "✦ Родинки и особенности: на месте\n"
            "💾 Размер файла: {size} MB\n\n"
            "<i>Ретушёр делал бы это {mins}–30 минут вручную.</i>"
        ),
        "en": (
            "📐 Resolution: original preserved\n"
            "🔍 Skin texture: preserved\n"
            "✦ Moles and features: intact\n"
            "💾 File size: {size} MB\n\n"
            "<i>A retoucher would spend {mins}–30 min doing this manually.</i>"
        ),
        "vi": (
            "📐 Độ phân giải: giữ nguyên bản gốc\n"
            "🔍 Kết cấu da: được bảo toàn\n"
            "✦ Nốt ruồi và đặc điểm: còn nguyên\n"
            "💾 Kích thước tệp: {size} MB\n\n"
            "<i>Chuyên gia retouch sẽ mất {mins}–30 phút làm thủ công.</i>"
        ),
        "ky": (
            "📐 Чечилиш: оригинал сакталды\n"
            "🔍 Тери текстурасы: сакталды\n"
            "✦ Меңдер жана өзгөчөлүктөр: ордунда\n"
            "💾 Файл өлчөмү: {size} MB\n\n"
            "<i>Ретушер муну {mins}–30 мүнөт кол менен жасайт эле.</i>"
        ),
    },
    "trial_remaining": {
        "ru": "🎁 Осталось бесплатных: <b>{n}</b>",
        "en": "🎁 Free retouches left: <b>{n}</b>",
        "vi": "🎁 Còn lại miễn phí: <b>{n}</b>",
        "ky": "🎁 Акысыз калды: <b>{n}</b>",
    },
    "trial_last": {
        "ru": "⚠️ <b>Осталась 1 бесплатная обработка</b>\n\nОформи подписку заранее 👇",
        "en": "⚠️ <b>1 free retouch left</b>\n\nGet a subscription in advance 👇",
        "vi": "⚠️ <b>Còn 1 lần retouch miễn phí</b>\n\nMua đăng ký trước 👇",
        "ky": "⚠️ <b>1 акысыз иштетүү калды</b>\n\nАлдын ала жазылыңыз 👇",
    },
    "trial_exceeded": {
        "ru": (
            "💎 <b>Бесплатные обработки использованы</b>\n\n"
            "💡 Ретушёр берёт 300–1500 сом за одно фото.\n"
            "Retouch Lab — 990 сом в месяц без ограничений.\n\n"
            "⏰ <b>Акция:</b> первый месяц за <b>799 сом</b>\n\n"
            "Выбери тариф 👇"
        ),
        "en": (
            "💎 <b>Free retouches used up</b>\n\n"
            "💡 A retoucher charges $5–20 per photo.\n"
            "Retouch Lab — unlimited for just $11/month.\n\n"
            "Choose a plan 👇"
        ),
        "vi": (
            "💎 <b>Đã dùng hết lần retouch miễn phí</b>\n\n"
            "💡 Chuyên gia retouch tính 200–500K/ảnh.\n"
            "Retouch Lab — không giới hạn chỉ $11/tháng.\n\n"
            "Chọn gói 👇"
        ),
        "ky": (
            "💎 <b>Акысыз иштетүүлөр бүттү</b>\n\n"
            "💡 Ретушер бир сүрөткө 300–1500 сом алат.\n"
            "Retouch Lab — айына 990 сомго чексиз.\n\n"
            "Тариф тандаңыз 👇"
        ),
    },
    "timeout_error": {
        "ru": (
            "⏱ <b>Обработка заняла слишком много времени.</b>\n\n"
            "Попробуйте ещё раз или выберите более лёгкий режим:\n"
            "🌿 Натуральная ретушь или ✨ Чистая кожа."
        ),
        "en": (
            "⏱ <b>Processing took too long.</b>\n\n"
            "Please try again or choose a lighter mode:\n"
            "🌿 Natural Retouch or ✨ Clean Skin."
        ),
        "vi": (
            "⏱ <b>Xử lý mất quá nhiều thời gian.</b>\n\n"
            "Vui lòng thử lại hoặc chọn chế độ nhẹ hơn:\n"
            "🌿 Retouch Tự Nhiên hoặc ✨ Da Sạch."
        ),
        "ky": (
            "⏱ <b>Иштетүү өтө көп убакыт алды.</b>\n\n"
            "Кайра байкап көрүңүз же жеңилирээк режим тандаңыз:\n"
            "🌿 Табигый ретушь же ✨ Таза тери."
        ),
    },
    "send_as_file": {
        "ru": (
            "⚠️ <b>Telegram сжимает обычные фото</b>\n\n"
            "Для сохранения полного качества отправь фото <b>файлом</b>:\n\n"
            "📎 Скрепка → <b>Файл</b> → выбери фото"
        ),
        "en": (
            "⚠️ <b>Telegram compresses regular photos</b>\n\n"
            "To keep full quality, send as <b>file</b>:\n\n"
            "📎 Paperclip → <b>File</b> → select photo"
        ),
        "vi": (
            "⚠️ <b>Telegram nén ảnh thông thường</b>\n\n"
            "Để giữ chất lượng đầy đủ, gửi dưới dạng <b>tệp</b>:\n\n"
            "📎 Kẹp giấy → <b>Tệp</b> → chọn ảnh"
        ),
        "ky": (
            "⚠️ <b>Telegram кадимки сүрөттөрдү сыгат</b>\n\n"
            "Толук сапатты сактоо үчүн <b>файл</b> катары жибериңиз:\n\n"
            "📎 Кысим → <b>Файл</b> → сүрөт тандаңыз"
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ПОДПИСКА
    # ══════════════════════════════════════════════════════════════════════════

    "sub_active": {
        "ru": (
            "💎 <b>Подписка активна</b>\n\n"
            "📅 Тариф: <b>{plan}</b>\n"
            "⏳ Действует до: <b>{date}</b>\n\n"
            "Отправляйте фото — всё работает ✨"
        ),
        "en": (
            "💎 <b>Subscription active</b>\n\n"
            "📅 Plan: <b>{plan}</b>\n"
            "⏳ Valid until: <b>{date}</b>\n\n"
            "Send your photos — everything works ✨"
        ),
        "vi": (
            "💎 <b>Đăng ký đang hoạt động</b>\n\n"
            "📅 Gói: <b>{plan}</b>\n"
            "⏳ Có hiệu lực đến: <b>{date}</b>\n\n"
            "Gửi ảnh của bạn — mọi thứ hoạt động ✨"
        ),
        "ky": (
            "💎 <b>Жазылуу активдүү</b>\n\n"
            "📅 Тариф: <b>{plan}</b>\n"
            "⏳ Чейин иштейт: <b>{date}</b>\n\n"
            "Сүрөттөрдү жибериңиз — баары иштейт ✨"
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ВЫБОР ЯЗЫКА
    # ══════════════════════════════════════════════════════════════════════════

    "choose_language": {
        "ru": "🌐 Выберите язык / Choose language:",
        "en": "🌐 Choose language:",
        "vi": "🌐 Chọn ngôn ngữ:",
        "ky": "🌐 Тилди тандаңыз:",
    },
    "language_set": {
        "ru": "✅ Язык установлен: Русский",
        "en": "✅ Language set: English",
        "vi": "✅ Đã chọn ngôn ngữ: Tiếng Việt",
        "ky": "✅ Тил коюлду: Кыргызча",
    },
    "support_text": {
        "ru": (
            "💬 <b>Поддержка Retouch Lab</b>\n\n"
            "По вопросам подписки и оплаты:\n"
            "<b>@linkiway_support</b>"
        ),
        "en": (
            "💬 <b>Retouch Lab Support</b>\n\n"
            "For subscription and payment questions:\n"
            "<b>@linkiway_support</b>"
        ),
        "vi": (
            "💬 <b>Hỗ trợ Retouch Lab</b>\n\n"
            "Về câu hỏi đăng ký và thanh toán:\n"
            "<b>@linkiway_support</b>"
        ),
        "ky": (
            "💬 <b>Retouch Lab колдоосу</b>\n\n"
            "Жазылуу жана төлөм боюнча:\n"
            "<b>@linkiway_support</b>"
        ),
    },
}


def t(key: str, lang: str = "ru", **kwargs) -> str:
    """
    Получить текст на нужном языке.
    Пример: t("welcome", lang="en")
    Пример с параметрами: t("trial_remaining", lang="ru", n=2)
    """
    lang = lang if lang in LANGUAGES else "ru"
    text = TEXTS.get(key, {}).get(lang) or TEXTS.get(key, {}).get("ru", key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text
