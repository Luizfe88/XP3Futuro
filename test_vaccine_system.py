#!/usr/bin/env python3
"""
Script de teste para o Sistema de Vacina (Evolução)
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_system import apply_vaccine, is_vaccinated, _vaccine_cache
import logging

# Configurar logging para teste
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_vaccine_system():
    """Testa o sistema de vacina completo"""
    print("\n🧪 Testando SISTEMA DE VACINA (EVOLUÇÃO)...")
    
    # Limpar cache antes do teste
    _vaccine_cache.clear()
    
    # Teste 1: Aplicar vacina
    print("\n1️⃣ Testando aplicação de vacina:")
    symbol = "WINZ26"
    reason = "Stop loss por slippage elevado"
    
    apply_vaccine(symbol, reason)
    print(f"   ✅ Vacina aplicada para {symbol}")
    print(f"   📋 Cache atual: {_vaccine_cache}")
    
    # Teste 2: Verificar se está vacinado
    print("\n2️⃣ Testando verificação de vacina:")
    is_vac = is_vaccinated(symbol)
    print(f"   🔍 {symbol} está vacinado? {is_vac}")
    
    # Teste 3: Verificar símbolo não vacinado
    print("\n3️⃣ Testando símbolo não vacinado:")
    other_symbol = "WDOZ26"
    is_vac_other = is_vaccinated(other_symbol)
    print(f"   🔍 {other_symbol} está vacinado? {is_vac_other}")
    
    # Teste 4: Testar expiração (simulação)
    print("\n4️⃣ Testando expiração de vacina:")
    # Simular expiração manualmente
    _vaccine_cache[symbol] = time.time() - 1  # Expirado
    is_vac_expired = is_vaccinated(symbol)
    print(f"   🔍 {symbol} ainda está vacinado após expiração? {is_vac_expired}")
    print(f"   📋 Cache após expiração: {_vaccine_cache}")
    
    # Teste 5: Testar diferentes razões
    print("\n5️⃣ Testando filtros de razão:")
    test_cases = [
        ("slippage", True),
        ("spread elevado", True),
        ("STOP LOSS TÉCNICO", False),
        ("Take profit atingido", False)
    ]
    
    for reason, should_apply in test_cases:
        _vaccine_cache.clear()
        apply_vaccine("TESTE", reason)
        applied = len(_vaccine_cache) > 0
        status = "✅" if applied == should_apply else "❌"
        print(f"   {status} Razão: '{reason}' - Vacina aplicada: {applied} (esperado: {should_apply})")

def main():
    """Executa todos os testes"""
    print("🚀 Iniciando testes do Sistema de Vacina")
    print("=" * 60)
    
    try:
        test_vaccine_system()
        print("\n✅ Todos os testes de vacina concluídos com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())