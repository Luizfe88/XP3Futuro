import MetaTrader5 as mt5
import config
import time

print(f"Conectando em: {config.MT5_TERMINAL_PATH}")

if not mt5.initialize(path=config.MT5_TERMINAL_PATH):
    print(f"❌ Falha ao conectar: {mt5.last_error()}")
    quit()

acc = mt5.account_info()
if acc:
    print("="*40)
    print(f"✅ CONECTADO")
    print(f"👤 Login:    {acc.login}")
    print(f"🏢 Server:   {acc.server}")
    print(f"💰 Balance:  {acc.balance:,.2f}")
    print(f"📈 Equity:   {acc.equity:,.2f}")
    print(f"💵 Profit:   {acc.profit:,.2f}")
    print("="*40)
else:
    print("❌ Falha ao obter dados da conta")

mt5.shutdown()
