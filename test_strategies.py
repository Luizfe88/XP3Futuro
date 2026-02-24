#!/usr/bin/env python3
"""
Script para testar se as estratégias VOLATILITY_BREAKOUT e MEAN_REVERSION estão funcionando corretamente
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimizer_optuna import tournament_report
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Criar dados de teste sintéticos
def create_test_data():
    """Cria dados de teste sintéticos para WIN$N"""
    np.random.seed(42)
    
    # Parâmetros do WIN$N (mini-índice)
    start_price = 120000  # Preço típico do WIN$N
    volatility = 0.02     # Volatilidade diária
    trend = 0.001         # Tendência ligeiramente positiva
    
    # Criar 1000 dias de dados (aproximadamente 4 anos)
    n_days = 1000
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Gerar preços com caminhada aleatória com tendência
    prices = [start_price]
    for i in range(1, n_days):
        daily_return = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Criar DataFrame com OHLCV
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # Gerar OHLC a partir dos preços de fechamento
    df['open'] = df['close'].shift(1)
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, n_days))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, n_days))
    df['volume'] = np.random.uniform(1000, 5000, n_days)
    
    # Remover primeira linha (sem dados de abertura)
    df = df.dropna()
    df.set_index('date', inplace=True)
    
    return df

def test_strategies():
    """Testa ambas as estratégias usando pytest."""
    print("🧪 Testando estratégias VOLATILITY_BREAKOUT e MEAN_REVERSION...")
    
    # Criar dados de teste
    df = create_test_data()
    print(f"✅ Dados de teste criados: {len(df)} dias")
    
    # Testar ambas as estratégias
    result = tournament_report("WIN$N", df, n_trials=5, timeout=300)
    
    print("\n📊 Resultados do torneio:")
    print(f"Símbolo: {result['symbol']}")
    print(f"Vencedor: {result['winner']}")
    
    assert result is not None
    assert result['symbol'] == "WIN$N"
    assert 'winner' in result
    assert result['winner'] != 'NONE'
    assert 'results' in result
    assert len(result['results']) > 0

    for r in result['results']:
        strategy = r['strategy']
        
        print(f"\n📈 {strategy}:")
        
        assert 'best_params' in r
        assert 'best_score' in r
        
        if r.get('tradeable'):
            print(f"  Tradeable: {r['tradeable']}")
            assert 'metrics' in r
            metrics = r['metrics']
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            assert metrics.get('total_trades', 0) > 0
        else:
            print("  Não negociável")

    print("\n✅ Teste concluído com sucesso!")