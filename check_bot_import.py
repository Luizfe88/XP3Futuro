import sys
import os

try:
    import xp3future as config
    print("DEBUG: Imported xp3future as config")
except ModuleNotFoundError:
    print("DEBUG: Module xp3future not found, adjusting path...")
    _sys = sys
    _os = os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    import xp3future as config
    print("DEBUG: Imported xp3future as config after path adjustment")

print(f"DEBUG: config.ENABLE_TELEGRAM_NOTIF = {getattr(config, 'ENABLE_TELEGRAM_NOTIF', 'NOT_FOUND')}")
print(f"DEBUG: config type: {type(config)}")
print(f"DEBUG: config file/path: {getattr(config, '__file__', 'NONE')}")
if hasattr(config, '__path__'):
    print(f"DEBUG: config path: {config.__path__}")
