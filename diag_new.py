import MetaTrader5 as mt5
import utils
import config
import sys
import logging

# Configura logger básico para ver os logs do utils
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

print("--- Iniciando Diagnóstico com Nova Lógica (utils.initialize_mt5) ---")

if utils.initialize_mt5():
    print("✅ SUCESSO: Conectado e Validado via utils.initialize_mt5()")
    
    # 2. Informações do servidor
    info = mt5.account_info()
    term = mt5.terminal_info()
    if info:
        print(f"Servidor: {info.server}")
        print(f"Broker  : {info.company}")
        print(f"Conta   : {info.login}")
    if term:
        print(f"Path do Terminal: {term.path}")
        print(f"Conectado: {term.connected}")
    
    # 3. Teste rápido de símbolos B3
    print("\n--- Teste de Símbolos B3 ---")
    for g in ["WDO$N", "WIN$N"]:
        sel = mt5.symbol_select(g, True)
        if sel:
            inf = mt5.symbol_info(g)
            print(f"✅ {g}: Selecionado (Visível: {inf.visible if inf else 'N/A'})")
        else:
            print(f"❌ {g}: Falha na seleção")
            
    mt5.shutdown()
else:
    print("❌ FALHA: Não foi possível conectar ao terminal correto da XP.")
    sys.exit(1)
