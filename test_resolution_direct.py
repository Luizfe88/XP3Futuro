import MetaTrader5 as mt5
import sys
import re

def print_flush(*args):
    print(*args)
    sys.stdout.flush()

print_flush("🚀 Iniciando Teste de Resolução Direta (Minimal Imports)...")

# 1. Conecta logo de cara
if mt5.initialize():
    print_flush("✅ MT5 Conectado (sem path).")
else:
    # Tenta com path se o padrão falhar
    path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
    print_flush(f"⚠️ Falha na conexão simples. Tentando com path: {path}...")
    if not mt5.initialize(path=path):
        print_flush(f"❌ Falha ao inicializar MT5: {mt5.last_error()}")
        sys.exit(1)
    print_flush("✅ MT5 Conectado (com path).")

# 2. Só agora importa o resto
try:
    print_flush("📦 Importando bibliotecas adicionais...")
    import utils
    from datetime import datetime
    import config_futures
    print_flush("✅ Importações concluídas.")

    bases = ['WIN', 'WDO', 'IND', 'DOL', 'ICF', 'CCM', 'BIT', 'DI1', 'WSP', 'BGI', 'SFI']
    print_flush(f"📋 Testando bases: {bases}")
    
    for base in bases:
        print_flush(f"\n🔍 Resolvendo base: {base}")
        # A nova lógica usa symbol_select dentro de get_futures_candidates
        candidates = utils.get_futures_candidates(base)
        
        if candidates:
            for cand in candidates:
                sym = cand.get('symbol')
                vol = cand.get('volume', 0)
                days = cand.get('days_to_exp', 0)
                print_flush(f"  ✅ Encontrado: {sym} (Volume: {vol:.0f}, Dias para expirar: {days})")
        else:
            print_flush(f"  ❌ Nenhum candidato encontrado para {base}")
            
    # Teste final com os padrões específicos do usuário
    patterns = ['WIN$N','IND$N','WSP$N','WDO$N','DOL$N','CCM$N','BGI$N','ICF$N','BIT$N','DI1$N']
    print_flush("\n📈 Testando padrões específicos do usuário via resolve_symbol:")
    for pat in patterns:
        res = utils.resolve_symbol(pat)
        if res:
            print_flush(f"  ✅ {pat} -> {res}")
        else:
            print_flush(f"  ❌ {pat} -> FALHA na resolução")

except Exception as e:
    print_flush(f"💥 Erro durante o teste: {e}")
    import traceback
    traceback.print_exc()
finally:
    mt5.shutdown()
    print_flush("\n🔌 MT5 desconectado.")
