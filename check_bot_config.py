import config
import os
print(f"DEBUG: os.getenv('ENABLE_TELEGRAM_NOTIF') = {os.getenv('ENABLE_TELEGRAM_NOTIF')}")
print(f"DEBUG: config.ENABLE_TELEGRAM_NOTIF = {getattr(config, 'ENABLE_TELEGRAM_NOTIF', 'NOT_FOUND')}")
print(f"DEBUG: config file path: {config.__file__}")
