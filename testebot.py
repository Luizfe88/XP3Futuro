import MetaTrader5 as mt5
import config

print("🔍 Teste 1: MT5 Connection")
if mt5.initialize():
    print("✅ MT5 conectado")
    print(f"   Account: {mt5.account_info().login}")
    mt5.shutdown()
else:
    print("❌ MT5 falhou - verifique se está rodando")

print("\n🔍 Teste 2: Telegram")
try:
    from utils import get_telegram_bot
    bot = get_telegram_bot()
    bot.send_message(config.TELEGRAM_CHAT_ID, "🧪 Teste de conexão OK!")
    print("✅ Telegram funcionando")
except Exception as e:
    print(f"❌ Telegram falhou: {e}")

print("\n🔍 Teste 3: Imports")
try:
    from bot import main
    print("✅ bot.py OK")
except Exception as e:
    print(f"❌ bot.py tem erro: {e}")