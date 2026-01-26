"""
TESTE PRÉ-OTIMIZAÇÃO - VERIFICAÇÃO COMPLETA
Execute este script antes de rodar o otimizador
Tempo: ~1 minuto
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

try:
    import config
except ImportError:
    config = None

print("="*80)
print("🔍 TESTE PRÉ-OTIMIZAÇÃO - VERIFICAÇÃO COMPLETA")
print("="*80)

# ===========================
# 1. VERIFICAÇÃO DO MT5
# ===========================
print("\n[1/5] 🔌 Verificando conexão MT5...")

if not mt5:
    print("❌ ERRO: Módulo MetaTrader5 não instalado")
    print("   Execute: pip install MetaTrader5")
    exit(1)

if not mt5.initialize():
    print("❌ ERRO: MT5 não inicializado")
    print("   • Abra o MetaTrader 5 manualmente")
    print("   • Faça login na sua conta")
    print("   • Aguarde conectar")
    exit(1)

terminal_info = mt5.terminal_info()
account_info = mt5.account_info()

print("✅ MT5 conectado")
print(f"   Login: {account_info.login}")
print(f"   Servidor: {account_info.server}")
print(f"   Corretora: {account_info.company}")
print(f"   Conectado: {terminal_info.connected}")

if not terminal_info.connected:
    print("\n❌ ERRO: Terminal não conectado ao servidor")
    print("   Clique no canto inferior do MT5 e faça login")
    mt5.shutdown()
    exit(1)

# ===========================
# 2. VERIFICAÇÃO DO CONFIG.PY
# ===========================
print("\n[2/5] 📝 Verificando config.py...")

if not config:
    print("❌ ERRO: config.py não encontrado")
    print("   Crie um arquivo config.py na mesma pasta")
    mt5.shutdown()
    exit(1)

SECTOR_MAP = getattr(config, "SECTOR_MAP", {})

if not SECTOR_MAP:
    print("❌ ERRO: SECTOR_MAP vazio ou não encontrado no config.py")
    print("\n   Adicione no config.py:")
    print("   SECTOR_MAP = {")
    print("       'PETR4': 'Petróleo',")
    print("       'VALE3': 'Mineração',")
    print("   }")
    mt5.shutdown()
    exit(1)

symbols_list = [k.upper().strip() for k in SECTOR_MAP.keys() if isinstance(k, str)]

print(f"✅ config.py carregado")
print(f"   SECTOR_MAP: {len(symbols_list)} símbolos")
print(f"   Primeiros 5: {', '.join(symbols_list[:5])}")

# ===========================
# 3. VALIDAÇÃO DOS SÍMBOLOS
# ===========================
print(f"\n[3/5] 🔍 Validando {len(symbols_list)} símbolos no MT5...")

valid_symbols = []
invalid_symbols = []

for symbol in symbols_list[:10]:  # Testa apenas os primeiros 10
    info = mt5.symbol_info(symbol)
    if info:
        valid_symbols.append(symbol)
        print(f"   ✅ {symbol:8} - {info.description[:40]}")
    else:
        invalid_symbols.append(symbol)
        print(f"   ❌ {symbol:8} - NÃO ENCONTRADO")

if invalid_symbols:
    print(f"\n⚠️ AVISO: {len(invalid_symbols)} símbolo(s) inválido(s):")
    for sym in invalid_symbols:
        print(f"   - {sym}")
    print("\n   AÇÃO NECESSÁRIA:")
    print("   1. No MT5, vá em Ctrl+U")
    print("   2. Procure cada símbolo inválido")
    print("   3. Anote o nome EXATO como aparece")
    print("   4. Atualize o SECTOR_MAP no config.py")

if not valid_symbols:
    print("\n❌ ERRO CRÍTICO: Nenhum símbolo válido!")
    print("   Verifique se o formato está correto:")
    print("   • Clear/XP: PETR4 (sem $ ou .SA)")
    print("   • Outras: Verifique no MT5 (Ctrl+U)")
    mt5.shutdown()
    exit(1)

print(f"\n✅ {len(valid_symbols)}/{len(symbols_list[:10])} símbolos válidos (amostra)")

# ===========================
# 4. TESTE DE CARREGAMENTO DE DADOS
# ===========================
print(f"\n[4/5] 📊 Testando carregamento de dados...")

test_symbol = valid_symbols[0]
print(f"   Símbolo de teste: {test_symbol}")

# Teste 1: copy_rates_from_pos (método principal)
print(f"\n   Método 1: copy_rates_from_pos")
rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_M15, 0, 100)

if rates is not None and len(rates) > 0:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"   ✅ SUCESSO! {len(rates)} barras carregadas")
    print(f"   📅 Período: {df['time'].min()} até {df['time'].max()}")
    print(f"   💰 Última cotação: R$ {rates[-1]['close']:.2f}")
else:
    error = mt5.last_error()
    print(f"   ❌ FALHOU: {error}")
    print(f"\n   Testando método alternativo...")
    
    # Teste 2: copy_rates_range
    from datetime import timedelta
    utc_to = datetime.now()
    utc_from = utc_to - timedelta(days=1)
    rates = mt5.copy_rates_range(test_symbol, mt5.TIMEFRAME_M15, utc_from, utc_to)
    
    if rates is not None and len(rates) > 0:
        print(f"   ✅ copy_rates_range funcionou! {len(rates)} barras")
    else:
        print(f"   ❌ Ambos os métodos falharam")
        print(f"\n   POSSÍVEIS CAUSAS:")
        print(f"   1. Fora do horário de mercado")
        print(f"   2. Dados históricos indisponíveis")
        print(f"   3. Conta sem permissão")

# Teste de performance (muitas barras)
print(f"\n   Teste de performance (20000 barras)...")
start = time.time()
rates_large = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_M15, 0, 20000)
elapsed = time.time() - start

if rates_large is not None and len(rates_large) > 0:
    print(f"   ✅ {len(rates_large)} barras em {elapsed:.2f}s ({len(rates_large)/elapsed:.0f} barras/s)")
    
    df_large = pd.DataFrame(rates_large)
    df_large['time'] = pd.to_datetime(df_large['time'], unit='s')
    days = (df_large['time'].max() - df_large['time'].min()).days
    print(f"   📅 Histórico: {days} dias de dados")
    
    if len(rates_large) >= 10000:
        print(f"   🎉 EXCELENTE! Dados suficientes para otimização robusta")
    elif len(rates_large) >= 5000:
        print(f"   ✅ BOM! Dados adequados para otimização")
    else:
        print(f"   ⚠️ AVISO: Poucos dados ({len(rates_large)} barras)")
else:
    print(f"   ⚠️ Não foi possível carregar dataset grande")

# ===========================
# 5. VERIFICAÇÃO DO MARKET WATCH
# ===========================
print(f"\n[5/5] 👁️  Verificando Market Watch...")

all_symbols = mt5.symbols_get()
if all_symbols:
    visible_count = len([s for s in all_symbols if s.visible])
    print(f"   Total de símbolos: {len(all_symbols)}")
    print(f"   Visíveis no Market Watch: {visible_count}")
    
    if visible_count >= 5000:
        print(f"   ⚠️ AVISO: Market Watch CHEIO ({visible_count}/5000)")
        print(f"   RECOMENDAÇÃO:")
        print(f"   1. Use: python sync_market_watch.py --clear")
        print(f"   2. OU use: python otimizador_clear_xp.py (não precisa de MW)")
    elif visible_count >= 1000:
        print(f"   ⚠️ Market Watch grande, considere limpar")
    else:
        print(f"   ✅ Market Watch OK")
    
    # Verifica quantos do SECTOR_MAP estão no MW
    sector_in_mw = [s.name for s in all_symbols if s.visible and s.name in symbols_list]
    print(f"   Do SECTOR_MAP no MW: {len(sector_in_mw)}/{len(symbols_list)}")

# ===========================
# RESUMO FINAL
# ===========================
print("\n" + "="*80)
print("📊 RESUMO DA VERIFICAÇÃO")
print("="*80)

checks = {
    "MT5 Conectado": terminal_info.connected if terminal_info else False,
    "config.py OK": bool(SECTOR_MAP),
    "Símbolos Válidos": len(valid_symbols) > 0,
    "Dados Carregam": rates is not None and len(rates) > 0,
    "Histórico Suficiente": rates_large is not None and len(rates_large) >= 5000,
}

all_ok = all(checks.values())

for check, status in checks.items():
    icon = "✅" if status else "❌"
    print(f"{icon} {check}")

print("\n" + "="*80)

if all_ok:
    print("🎉 TUDO PRONTO PARA OTIMIZAÇÃO!")
    print("\n💡 PRÓXIMOS PASSOS:")
    print("   1. python sync_market_watch.py --clear  (limpar MW)")
    print("   2. python otimizador_auto_sync.py       (otimizar)")
    print("\n   OU diretamente:")
    print("   python otimizador_clear_xp.py           (não usa MW)")
    
elif checks["MT5 Conectado"] and checks["Símbolos Válidos"] and checks["Dados Carregam"]:
    print("✅ PRONTO! (com pequenos avisos)")
    print("\n💡 Você pode prosseguir com a otimização")
    print("   Execute: python otimizador_clear_xp.py")
    
else:
    print("❌ PROBLEMAS ENCONTRADOS - CORRIJA ANTES DE OTIMIZAR")
    print("\n🔧 AÇÕES NECESSÁRIAS:")
    
    if not checks["MT5 Conectado"]:
        print("   • Abra o MT5 e faça login")
    
    if not checks["config.py OK"]:
        print("   • Crie/corrija o config.py com SECTOR_MAP")
    
    if not checks["Símbolos Válidos"]:
        print("   • Corrija os símbolos no SECTOR_MAP")
        print("   • Use formato: PETR4 (sem $ ou .SA)")
    
    if not checks["Dados Carregam"]:
        print("   • Verifique horário de mercado")
        print("   • Abra gráfico do símbolo no MT5 manualmente")

print("="*80)

# Cleanup
mt5.shutdown()
print("\n✅ MT5 desconectado. Teste concluído!\n")



# === TESTE RÁPIDO: LIMITAR A 10 SÍMBOLOS ===
    test_symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4", "PRIO3", 
                    "VBBR3", "SUZB3", "WEGE3", "ABEV3", "EQTL3"]  # Escolha os que quiser
    symbols_to_optimize = [s for s in symbols_to_optimize if s in test_symbols]
    # Ou simplesmente: symbols_to_optimize = test_symbols[:10]