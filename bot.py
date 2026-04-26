import asyncio
import logging
import os
import requests

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from aiogram.filters import Command
from dotenv import load_dotenv

# загрузка токена
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

RUNPOD_URL = "https://ychsinzriqqmll-8000.proxy.runpod.net/process-image"


# ГЛАВНОЕ МЕНЮ
main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📸 Обработать фото")],
        [KeyboardButton(text="✨ Улучшить качество")],
        [KeyboardButton(text="💎 Подписка")],
        [KeyboardButton(text="ℹ️ О боте")]
    ],
    resize_keyboard=True
)

# КНОПКА НАЗАД
back_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="⬅️ Назад")]
    ],
    resize_keyboard=True
)


# СТАРТ
@dp.message(Command("start"))
async def start(message: Message):

    text = """
Привет! 👋

Я *Retouch Lab* — AI бот для профессиональной обработки фото.

Моя цель — сэкономить ваше время на ретуши.

Что я умею:

✨ выравнивать свет  
🎨 улучшать цвет  
🧴 делать natural skin tone  
📷 повышать качество изображения  

Выберите действие ниже 👇
"""

    await message.answer(text, reply_markup=main_menu, parse_mode="Markdown")


# ОБРАБОТАТЬ ФОТО
@dp.message(F.text == "📸 Обработать фото")
async def process_photo(message: Message):

    text = """
📸 Обработка фото

Отправьте фотографию ФАЙЛОМ.

Подходят:
• портреты  
• fashion  
• семейные фото  

Формат:

• JPEG  
• iPhone  
• Android  

Что я сделаю:

✨ выровняю свет  
🎨 улучшу цвет  
🧴 natural skin tone  
📷 повышу качество  

Жду фото 📸
"""

    await message.answer(text, reply_markup=back_menu, parse_mode="Markdown")


# УЛУЧШИТЬ КАЧЕСТВО
@dp.message(F.text == "✨ Улучшить качество")
async def enhance_photo(message: Message):

    text = """
✨ Улучшение качества

AI может восстановить:

• старые фотографии  
• размытые фото  
• фото низкого качества  

Что будет улучшено:

📷 резкость  
🎨 цвет  
🔍 детали  

Отправьте фотографию ФАЙЛОМ.
"""

    await message.answer(text, reply_markup=back_menu, parse_mode="Markdown")


# ПОДПИСКА
@dp.message(F.text == "💎 Подписка")
async def subscription(message: Message):

    text = """
💎 Подписка Retouch Lab

Тарифы:

Starter  
50 фото / месяц

Pro  
300 фото / месяц

Studio  
Безлимит

Скоро появится подключение оплаты.
"""

    await message.answer(text, reply_markup=back_menu)


# О БОТЕ
@dp.message(F.text == "ℹ️ О боте")
async def about(message: Message):

    text = """
ℹ️ О Retouch Lab

Retouch Lab — AI инструмент для обработки фотографий.

Мы создали его чтобы сэкономить время:

📸 фотографам  
🎥 блогерам  
🧑‍💻 креаторам  

AI автоматически:

✨ выравнивает свет  
🎨 улучшает цвет  
🧴 делает natural skin tone  
📷 повышает качество  

То, что раньше занимало 20 минут ретуши — теперь занимает несколько секунд.
"""

    await message.answer(text, reply_markup=back_menu)


# КНОПКА НАЗАД
@dp.message(F.text == "⬅️ Назад")
async def back(message: Message):
    await message.answer("Главное меню 👇", reply_markup=main_menu)


# ПРИЕМ ФОТО ФАЙЛОМ
@







import asyncio
import logging
import os
import requests

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from aiogram.filters import Command
from dotenv import load_dotenv

# загрузка токена
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

RUNPOD_URL = "https://ychsinzriqqmll-8000.proxy.runpod.net/process-image"


# ГЛАВНОЕ МЕНЮ
main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📸 Обработать фото")],
        [KeyboardButton(text="✨ Улучшить качество")],
        [KeyboardButton(text="💎 Подписка")],
        [KeyboardButton(text="ℹ️ О боте")]
    ],
    resize_keyboard=True
)

# КНОПКА НАЗАД
back_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="⬅️ Назад")]
    ],
    resize_keyboard=True
)


# СТАРТ
@dp.message(Command("start"))
async def start(message: Message):

    text = """
Привет! 👋

Я *Retouch Lab* — AI бот для профессиональной обработки фото.

Моя цель — сэкономить ваше время на ретуши.

Что я умею:

✨ выравнивать свет  
🎨 улучшать цвет  
🧴 делать natural skin tone  
📷 повышать качество изображения  

Выберите действие ниже 👇
"""

    await message.answer(text, reply_markup=main_menu, parse_mode="Markdown")


# ОБРАБОТАТЬ ФОТО
@dp.message(F.text == "📸 Обработать фото")
async def process_photo(message: Message):

    text = """
📸 Обработка фото

Отправьте фотографию ФАЙЛОМ.

Подходят:
• портреты  
• fashion  
• семейные фото  

Формат:

• JPEG  
• iPhone  
• Android  

Что я сделаю:

✨ выровняю свет  
🎨 улучшу цвет  
🧴 natural skin tone  
📷 повышу качество  

Жду фото 📸
"""

    await message.answer(text, reply_markup=back_menu, parse_mode="Markdown")


# УЛУЧШИТЬ КАЧЕСТВО
@dp.message(F.text == "✨ Улучшить качество")
async def enhance_photo(message: Message):

    text = """
✨ Улучшение качества

AI может восстановить:

• старые фотографии  
• размытые фото  
• фото низкого качества  

Что будет улучшено:

📷 резкость  
🎨 цвет  
🔍 детали  

Отправьте фотографию ФАЙЛОМ.
"""

    await message.answer(text, reply_markup=back_menu, parse_mode="Markdown")


# ПОДПИСКА
@dp.message(F.text == "💎 Подписка")
async def subscription(message: Message):

    text = """
💎 Подписка Retouch Lab

Тарифы:

Starter  
50 фото / месяц

Pro  
300 фото / месяц

Studio  
Безлимит

Скоро появится подключение оплаты.
"""

    await message.answer(text, reply_markup=back_menu)


# О БОТЕ
@dp.message(F.text == "ℹ️ О боте")
async def about(message: Message):

    text = """
ℹ️ О Retouch Lab

Retouch Lab — AI инструмент для обработки фотографий.

Мы создали его чтобы сэкономить время:

📸 фотографам  
🎥 блогерам  
🧑‍💻 креаторам  

AI автоматически:

✨ выравнивает свет  
🎨 улучшает цвет  
🧴 делает natural skin tone  
📷 повышает качество  

То, что раньше занимало 20 минут ретуши — теперь занимает несколько секунд.
"""

    await message.answer(text, reply_markup=back_menu)


# КНОПКА НАЗАД
@dp.message(F.text == "⬅️ Назад")
async def back(message: Message):
    await message.answer("Главное меню 👇", reply_markup=main_menu)


# ПРИЕМ ФОТО ФАЙЛОМ
@dp.message(F.document)
async def get_photo(message: Message):

    await message.answer("Фото получено 📸\nНачинаю AI обработку...")

    file = await bot.get_file(message.document.file_id)
    file_path = file.file_path

    downloaded_file = await bot.download_file(file_path)

    with open("input.jpg", "wb") as new_file:
        new_file.write(downloaded_file.read())

    # отправка в RunPod
    with open("input.jpg", "rb") as f:
        response = requests.post(
            RUNPOD_URL,
            files={"file": f}
        )

    if response.status_code != 200:
        await message.answer("Ошибка обработки 😔")
        return

    with open("output.jpg", "wb") as f:
        f.write(response.content)

    photo = FSInputFile("output.jpg")

    await message.answer_photo(photo, caption="✨ Готово! AI улучшил качество фото.")


# ЗАПУСК БОТА
async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
