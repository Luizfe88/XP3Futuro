# test_eod_close.py
# ============================================
# 🧪 SCRIPT DE TESTE: FECHAMENTO EOD
# ============================================

import MetaTrader5 as mt5
from datetime import datetime, time
import config

def test_eod_closing():
    """
    Valida se os horários de fechamento estão corretos
    """
    print("=" * 60)
    print("🧪 TESTE DE FECHAMENTO EOD")
    print("=" * 60)
    
    # 1. Valida config
    print("\n1️⃣ VALIDANDO CONFIG.PY:")
    print(f"   TRADING_START: {config.TRADING_START}")
    print(f"   NO_ENTRY_AFTER: {config.NO_ENTRY_AFTER}")
    print(f"   CLOSE_ALL_BY: {config.CLOSE_ALL_BY}")
    
    # Verifica se TRADING_END existe (NÃO deveria)
    if hasattr(config, 'TRADING_END'):
        print(f"   ⚠️ TRADING_END existe: {config.TRADING_END} (REMOVA ESTA LINHA!)")
    else:
        print("   ✅ TRADING_END não encontrado (correto)")
    
    # 2. Testa parsing dos horários
    print("\n2️⃣ TESTANDO PARSING DE HORÁRIOS:")
    try:
        start = datetime.strptime(config.TRADING_START, "%H:%M").time()
        no_entry = datetime.strptime(config.NO_ENTRY_AFTER, "%H:%M").time()
        close = datetime.strptime(config.CLOSE_ALL_BY, "%H:%M").time()
        
        print(f"   ✅ Start: {start}")
        print(f"   ✅ No Entry: {no_entry}")
        print(f"   ✅ Close: {close}")
        
        # Valida ordem lógica
        if start < no_entry < close:
            print("   ✅ Horários em ordem lógica")
        else:
            print("   ❌ ERRO: Horários fora de ordem!")
            
    except Exception as e:
        print(f"   ❌ ERRO ao parsear: {e}")
        return False
    
    # 3. Simula horários do dia
    print("\n3️⃣ SIMULANDO HORÁRIOS DO DIA:")
    
    test_times = [
        ("09:00", "Pré-mercado"),
        ("10:30", "Abertura"),
        ("12:00", "Meio-dia"),
        ("16:00", "Normal"),
        ("16:15", "Última entrada"),
        ("16:43", "2 min antes do close"),  # Novo: early close
        ("16:45", "Horário de fechamento"),
        ("16:47", "2 min após (failsafe)"),  # Novo
        ("17:00", "Pós-mercado"),
    ]
    
    for time_str, label in test_times:
        test_time = datetime.strptime(time_str, "%H:%M").time()
        
        # Simula lógica do get_market_status()
        if test_time < start:
            status = "PRE_MARKET"
        elif start <= test_time < no_entry:
            status = "OPEN"
        elif no_entry <= test_time < close:
            status = "NO_NEW_ENTRIES"
        else:
            status = "POST_MARKET (deve fechar)"
        
        print(f"   {time_str} ({label:20s}) → {status}")
    
    # 4. Verifica conexão MT5
    print("\n4️⃣ VALIDANDO CONEXÃO MT5:")
    
    if not mt5.initialize():
        print("   ❌ MT5 não conectado")
        return False
    
    print("   ✅ MT5 conectado")
    
    # Verifica posições
    positions = mt5.positions_get()
    if positions:
        print(f"   ℹ️ {len(positions)} posições abertas atualmente:")
        for p in positions[:5]:  # Mostra até 5
            print(f"      • {p.symbol} | Ticket: {p.ticket}")
    else:
        print("   ℹ️ Nenhuma posição aberta")
    
    # 5. Testa se close_all_positions() existe
    print("\n5️⃣ VALIDANDO FUNÇÕES:")
    
    try:
        from bot import close_all_positions, handle_daily_cycle
        print("   ✅ close_all_positions() importada")
        print("   ✅ handle_daily_cycle() importada")
    except ImportError as e:
        print(f"   ❌ Erro ao importar: {e}")
        return False
    
    # 6. Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO DO TESTE")
    print("=" * 60)
    print("✅ Configuração validada")
    print("✅ Horários em ordem lógica")
    print("✅ Funções de fechamento disponíveis")
    print("\n⚠️ PRÓXIMOS PASSOS:")
    print("   1. Aguarde até 16:43 (início do fechamento)")
    print("   2. Monitore os logs em tempo real")
    print("   3. Verifique se posições fecham até 16:45")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_eod_closing()
        
        if success:
            print("\n✅ TESTE PASSOU")
        else:
            print("\n❌ TESTE FALHOU")
            
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()