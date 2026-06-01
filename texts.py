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

# ══════════════════════════════════════════════════════════════════════════════
# ДОПОЛНИТЕЛЬНЫЕ ТЕКСТЫ
# ══════════════════════════════════════════════════════════════════════════════

TEXTS.update({

    "about_full": {
        "ru": (
            "✨ <b>RETOUCH LAB</b> — AI-ретушёр от Linkiway\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎯 <b>Что делает бот:</b>\n\n"
            "✦ Убирает прыщи и дефекты кожи\n"
            "✦ Выравнивает тон кожи\n"
            "✦ Dodge & Burn — свет и тени лица\n"
            "✦ Улучшает глаза и зубы\n"
            "✦ Добавляет объём лицу\n"
            "✦ Сохраняет натуральность\n\n"
            "❌ Бот НЕ меняет:\n"
            "цвет фото, фон, одежду, волосы, губы\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 режимов:</b>\n\n"
            "✨ <b>Чистая кожа</b> — минимально, натурально\n"
            "🌿 <b>Натуральная ретушь</b> — для каждого дня\n"
            "💫 <b>Объём и свет</b> — глубина, светотень\n"
            "💄 <b>Beauty Pro</b> — для соцсетей\n"
            "🌟 <b>Журнальный стиль</b> — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📐 Оригинальное разрешение 4K / 24MP\n"
            "🎁 Первые 3 фото — бесплатно"
        ),
        "en": (
            "✨ <b>RETOUCH LAB</b> — AI retoucher by Linkiway\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎯 <b>What the bot does:</b>\n\n"
            "✦ Removes pimples and skin blemishes\n"
            "✦ Evens out skin tone\n"
            "✦ Dodge & Burn — light and shadow\n"
            "✦ Enhances eyes and teeth\n"
            "✦ Adds face volume\n"
            "✦ Keeps natural look\n\n"
            "❌ Bot does NOT change:\n"
            "photo color, background, clothes, hair, lips\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 modes:</b>\n\n"
            "✨ <b>Clean Skin</b> — minimal, natural\n"
            "🌿 <b>Natural Retouch</b> — for everyday\n"
            "💫 <b>Depth & Light</b> — volume, shadows\n"
            "💄 <b>Beauty Pro</b> — for social media\n"
            "🌟 <b>Magazine Style</b> — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📐 Original resolution 4K / 24MP\n"
            "🎁 First 3 photos — free"
        ),
        "vi": (
            "✨ <b>RETOUCH LAB</b> — AI retouch bởi Linkiway\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎯 <b>Bot làm gì:</b>\n\n"
            "✦ Xóa mụn và khuyết điểm da\n"
            "✦ Cân bằng tông màu da\n"
            "✦ Dodge & Burn — ánh sáng và bóng tối\n"
            "✦ Làm đẹp mắt và răng\n"
            "✦ Thêm khối lượng cho khuôn mặt\n"
            "✦ Giữ vẻ tự nhiên\n\n"
            "❌ Bot KHÔNG thay đổi:\n"
            "màu ảnh, nền, quần áo, tóc, môi\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 chế độ:</b>\n\n"
            "✨ <b>Da Sạch</b> — tối giản, tự nhiên\n"
            "🌿 <b>Retouch Tự Nhiên</b> — dùng hàng ngày\n"
            "💫 <b>Chiều Sâu & Ánh Sáng</b> — khối lượng\n"
            "💄 <b>Beauty Pro</b> — cho mạng xã hội\n"
            "🌟 <b>Phong Cách Tạp Chí</b> — editorial\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📐 Độ phân giải gốc 4K / 24MP\n"
            "🎁 3 ảnh đầu tiên — miễn phí"
        ),
        "ky": (
            "✨ <b>RETOUCH LAB</b> — Linkiway AI ретушёру\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎯 <b>Бот эмне кылат:</b>\n\n"
            "✦ Прыщтарды жана тери кемчиликтерин жок кылат\n"
            "✦ Тери тонун теңдейт\n"
            "✦ Dodge & Burn — жарык жана көлөкө\n"
            "✦ Көздөрдү жана тиштерди жакшыртат\n"
            "✦ Жүзгө көлөм кошот\n"
            "✦ Табигыйлыкты сактайт\n\n"
            "❌ Бот ӨЗГӨРТПӨЙТ:\n"
            "сүрөт түсүн, фонду, кийимди, чачты, эриндерди\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🎨 <b>5 режим:</b>\n\n"
            "✨ <b>Таза тери</b> — минималдуу, табигый\n"
            "🌿 <b>Табигый ретушь</b> — күн сайын\n"
            "💫 <b>Көлөм жана жарык</b> — тереңдик\n"
            "💄 <b>Beauty Pro</b> — социалдык тармактар\n"
            "🌟 <b>Журнал стили</b> — editorial look\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📐 Оригинал сапат 4K / 24MP\n"
            "🎁 Алгачкы 3 сүрөт — акысыз"
        ),
    },

    "send_photo_prompt": {
        "ru": "📎 Отправьте фото <b>файлом</b> (скрепка → Файл)\nФорматы: JPG · PNG · HEIC · WebP",
        "en": "📎 Send photo as <b>file</b> (paperclip → File)\nFormats: JPG · PNG · HEIC · WebP",
        "vi": "📎 Gửi ảnh dưới dạng <b>tệp</b> (kẹp giấy → Tệp)\nĐịnh dạng: JPG · PNG · HEIC · WebP",
        "ky": "📎 Сүрөттү <b>файл</b> катары жибериңиз\nФорматтар: JPG · PNG · HEIC · WebP",
    },

    "processing_msg": {
        "ru": "⏳ Обрабатываю фото...",
        "en": "⏳ Processing photo...",
        "vi": "⏳ Đang xử lý ảnh...",
        "ky": "⏳ Сүрөт иштелип жатат...",
    },

    "not_image": {
        "ru": "Пожалуйста, отправь изображение (JPEG, PNG, HEIC, WebP).",
        "en": "Please send an image (JPEG, PNG, HEIC, WebP).",
        "vi": "Vui lòng gửi ảnh (JPEG, PNG, HEIC, WebP).",
        "ky": "Сүрөт жибериңиз (JPEG, PNG, HEIC, WebP).",
    },

    "photo_blocked": {
        "ru": (
            "⚠️ <b>Telegram сжимает обычные фото</b>\n\n"
            "Для полного качества отправь фото <b>файлом</b>:\n"
            "📎 Скрепка → <b>Файл</b> → выбери фото"
        ),
        "en": (
            "⚠️ <b>Telegram compresses regular photos</b>\n\n"
            "For full quality send as <b>file</b>:\n"
            "📎 Paperclip → <b>File</b> → select photo"
        ),
        "vi": (
            "⚠️ <b>Telegram nén ảnh thông thường</b>\n\n"
            "Để giữ chất lượng đầy đủ gửi dưới dạng <b>tệp</b>:\n"
            "📎 Kẹp giấy → <b>Tệp</b> → chọn ảnh"
        ),
        "ky": (
            "⚠️ <b>Telegram кадимки сүрөттөрдү сыгат</b>\n\n"
            "Толук сапат үчүн <b>файл</b> катары жибериңиз:\n"
            "📎 Кысым → <b>Файл</b> → сүрөт тандаңыз"
        ),
    },

    "screenshot_received": {
        "ru": "✅ <b>Скриншот получен!</b>\n\nПроверяем оплату — подписка активируется\nв течение нескольких минут 💎",
        "en": "✅ <b>Screenshot received!</b>\n\nChecking payment — subscription will be activated\nwithin a few minutes 💎",
        "vi": "✅ <b>Đã nhận ảnh chụp màn hình!</b>\n\nĐang kiểm tra thanh toán — đăng ký sẽ được kích hoạt\ntrong vài phút 💎",
        "ky": "✅ <b>Скриншот алынды!</b>\n\nТөлөм текшерилүүдө — жазылуу\nбир нече мүнөттөн кийин активдешет 💎",
    },

    "sub_plans": {
        "ru": (
            "💎 <b>Подписка Retouch Lab</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📅 1 месяц — <b>990 сом</b> (~$11)\n"
            "📅 3 месяца — <b>2 490 сом</b> (~$28) · -15%\n"
            "📅 6 месяцев — <b>4 990 сом</b> (~$57) · -25%\n"
            "📅 1 год — <b>8 990 сом</b> (~$102) · -35% 🔥\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "Выберите тариф 👇"
        ),
        "en": (
            "💎 <b>Retouch Lab Subscription</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📅 1 month — <b>$11</b>\n"
            "📅 3 months — <b>$28</b> · -15%\n"
            "📅 6 months — <b>$57</b> · -25%\n"
            "📅 1 year — <b>$102</b> · -35% 🔥\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "💎 Payment via USDT (TRC20)\n\n"
            "Choose a plan 👇"
        ),
        "vi": (
            "💎 <b>Đăng ký Retouch Lab</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📅 1 tháng — <b>280,000 VND</b> (~$11)\n"
            "📅 3 tháng — <b>710,000 VND</b> (~$28) · -15%\n"
            "📅 6 tháng — <b>1,450,000 VND</b> (~$57) · -25%\n"
            "📅 1 năm — <b>2,600,000 VND</b> (~$102) · -35% 🔥\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🏦 Thanh toán qua Vietcombank\n\n"
            "Chọn gói 👇"
        ),
        "ky": (
            "💎 <b>Retouch Lab жазылуусу</b>\n\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "📅 1 ай — <b>990 сом</b> (~$11)\n"
            "📅 3 ай — <b>2 490 сом</b> (~$28) · -15%\n"
            "📅 6 ай — <b>4 990 сом</b> (~$57) · -25%\n"
            "📅 1 жыл — <b>8 990 сом</b> (~$102) · -35% 🔥\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "Тариф тандаңыз 👇"
        ),
    },

    "examples_updating": {
        "ru": "🎥 Примеры временно обновляются.",
        "en": "🎥 Examples are temporarily updating.",
        "vi": "🎥 Ví dụ đang được cập nhật.",
        "ky": "🎥 Мисалдар жаңыртылууда.",
    },

    "main_menu_title": {
        "ru": "Главное меню 👇",
        "en": "Main menu 👇",
        "vi": "Menu chính 👇",
        "ky": "Башкы меню 👇",
    },

    "choose_mode": {
        "ru": "Выберите режим обработки 👇",
        "en": "Choose processing mode 👇",
        "vi": "Chọn chế độ xử lý 👇",
        "ky": "Иштетүү режимин тандаңыз 👇",
    },

    "mode_selected": {
        "ru": "✅ <b>Режим выбран: {name}</b>\n\n{desc}\n\n📎 Отправьте фото файлом\nФорматы: JPG · PNG · HEIC · WebP",
        "en": "✅ <b>Mode selected: {name}</b>\n\n{desc}\n\n📎 Send photo as file\nFormats: JPG · PNG · HEIC · WebP",
        "vi": "✅ <b>Đã chọn chế độ: {name}</b>\n\n{desc}\n\n📎 Gửi ảnh dưới dạng tệp\nĐịnh dạng: JPG · PNG · HEIC · WebP",
        "ky": "✅ <b>Режим тандалды: {name}</b>\n\n{desc}\n\n📎 Сүрөттү файл катары жибериңиз\nФорматтар: JPG · PNG · HEIC · WebP",
    },

    "current_mode": {
        "ru": "✅ <b>Текущий режим: {name}</b>\n\n{desc}\n\nВыберите режим или отправьте фото 📎",
        "en": "✅ <b>Current mode: {name}</b>\n\n{desc}\n\nChoose mode or send photo 📎",
        "vi": "✅ <b>Chế độ hiện tại: {name}</b>\n\n{desc}\n\nChọn chế độ hoặc gửi ảnh 📎",
        "ky": "✅ <b>Учурдагы режим: {name}</b>\n\n{desc}\n\nРежим тандаңыз же сүрөт жибериңиз 📎",
    },

    "trial_free_remaining": {
        "ru": "🎁 <b>Осталось бесплатных: {n} из {total}</b>\n\n✅ Текущий режим: {name}\n\nВыберите режим или отправьте фото 📎",
        "en": "🎁 <b>Free retouches left: {n} of {total}</b>\n\n✅ Current mode: {name}\n\nChoose mode or send photo 📎",
        "vi": "🎁 <b>Còn lại miễn phí: {n}/{total}</b>\n\n✅ Chế độ hiện tại: {name}\n\nChọn chế độ hoặc gửi ảnh 📎",
        "ky": "🎁 <b>Акысыз калды: {n}/{total}</b>\n\n✅ Учурдагы режим: {name}\n\nРежим тандаңыз же сүрөт жибериңиз 📎",
    },

    "choose_lang": {
        "ru": "🌐 <b>Выберите язык:</b>",
        "en": "🌐 <b>Choose language:</b>",
        "vi": "🌐 <b>Chọn ngôn ngữ:</b>",
        "ky": "🌐 <b>Тилди тандаңыз:</b>",
    },

})
