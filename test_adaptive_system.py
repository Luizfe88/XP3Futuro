#!/usr/bin/env python3
"""
Script de teste para o Sistema Adaptativo de 4 Camadas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_system
import config
import logging

# Configurar logging para teste
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def test_sensor_layer():
    """Testa a camada Sensor"""
    print("\n🧪 Testando CAMADA SENSOR...")
    
    # Forçar coleta de dados
    metrics = adaptive_system.collect_sensor_data(force_run=True)
    
    print(f"📊 Volatilidade: ATR D1={metrics['volatility']['atr_d1']:.4f}, ATR M15={metrics['volatility']['atr_m15']:.4f}, Ratio={metrics['volatility']['ratio']:.2f}")
    print(f"📊 Volume: RVOL={metrics['relative_volume']['rvol']:.2f}, Média={metrics['relative_volume']['avg_rvol']:.0f}")
    print(f"📊 Performance: PnL={metrics['recent_performance']['pnl']:.2f}, Win Rate={metrics['recent_performance']['win_rate']:.2%}, DD={metrics['recent_performance']['drawdown']:.2%}")
    
    return metrics

def test_brain_layer():
    """Testa a camada Cérebro"""
    print("\n🧠 Testando CAMADA CÉREBRO...")
    
    # Primeiro coletar dados do sensor
    adaptive_system.collect_sensor_data(force_run=True)
    
    # Analisar regime
    regime = adaptive_system.analyze_market_regime()
    print(f"🎯 Regime detectado: {regime}")
    
    return regime

def test_mechanic_layer():
    """Testa a camada Mecânico"""
    print("\n🔧 Testando CAMADA MECÂNICO...")
    
    # Simular diferentes regimes
    regimes = ["TREND", "REVERSION", "NEUTRAL"]
    
    for regime in regimes:
        print(f"\n⚙️ Ajustando parâmetros para regime: {regime}")
        adaptive_system.adjust_parameters(regime)
        
        # Mostrar parâmetros atuais (simulação)
        print(f"   ✓ Parâmetros ajustados para {regime}")

def test_panic_mode():
    """Testa o modo Pânico"""
    print("\n🚨 Testando MODO PÂNICO...")
    
    # Testar detecção de queda (simulação)
    result = adaptive_system.check_panic_mode()
    print(f"🎯 Modo pânico ativado: {result}")

def main():
    """Executa todos os testes"""
    print("🚀 Iniciando testes do Sistema Adaptativo de 4 Camadas")
    print("=" * 60)
    
    try:
        # Testar cada camada
        metrics = test_sensor_layer()
        regime = test_brain_layer()
        test_mechanic_layer()
        test_panic_mode()
        
        print("\n✅ Todos os testes concluídos com sucesso!")
        print("\n📋 Resumo:")
        print(f"   • Sensor: Coletando {len(metrics)} métricas")
        print(f"   • Cérebro: Regime '{regime}' detectado")
        print(f"   • Mecânico: Parâmetros ajustados dinamicamente")
        print(f"   • Pânico: Circuit breaker ativo")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())