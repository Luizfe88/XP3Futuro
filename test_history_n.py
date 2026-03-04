import MetaTrader5 as mt5
import sys
import utils

def pf(*a): print(*a); sys.stdout.flush()

pf("🚀 Iniciando Teste de Resolução de Histórico Otimizado...")

if not mt5.initialize():
    pf("❌ Falha na conexão MT5")
    sys.exit(1)
pf("✅ MT5 Conectado.")

symbols = ["WIN", "WDO"]

possibilities = [
    "WINJ25", "WINJ26", "WINM25", "WINM26",
    "WDOJ25", "WDOJ26", "WDOK25", "WDOK26"
]
pf("\n🔍 Testando disponibilidade de símbolos explícitos:")
for p in possibilities:
    # Tenta selecionar
    mt5.symbol_select(p, True)
    info = mt5.symbol_info(p)
    if info is not None:
        pf(f"  🟢 {p} EXISTE. Tentando puxar candles...")
        df = utils.safe_copy_rates(p, mt5.TIMEFRAME_M5, 50)
        if df is not None:
            pf(f"     ✅ {len(df)} candles retornados para {p} do MT5.")
        else:
            pf(f"     ❌ Símbolo existe, mas SEM DADOS (df=None). fallback ativado talvez.")
    else:
        pf(f"  🔴 {p} NÃO EXISTE NO MT5 DA XP.")

mt5.shutdown()
pf("\n🔌 MT5 desconectado.")
