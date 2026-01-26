# telegram_async.py
import asyncio
from telegram import Bot
import config

_bot = None

def _get_bot():
    global _bot
    if _bot is None:
        _bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    return _bot

async def _send(chat_id: str, text: str):
    bot = _get_bot()
    await bot.send_message(chat_id=chat_id, text=text)

def send_async(chat_id: str, text: str):
    """
    Envio seguro de Telegram:
    - Streamlit-safe
    - Thread-safe
    - Python 3.10 → 3.13
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send(chat_id, text))
    except RuntimeError:
        asyncio.run(_send(chat_id, text))
