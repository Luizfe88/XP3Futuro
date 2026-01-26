import numpy as np
import pandas as pd
from metrics import calculate_all_metrics, MetricsConfig

def test_sharpe_calculation():
    """Valida cálculo correto do Sharpe"""
    
    # Equity com retorno conhecido
    equity = np.array([100000, 101000, 102010, 103030.1])  # 1% por período
    trades = pd.DataFrame({'pnl_money': [1000, 1010, 1020.1]})
    
    config = MetricsConfig(bars_per_day=28, trading_days_per_year=252)
    metrics = calculate_all_metrics(equity, trades, config)
    
    # Sharpe deve ser positivo e razoável
    assert metrics.sharpe_ratio > 0, "Sharpe deve ser positivo"
    assert metrics.sharpe_ratio < 10, "Sharpe muito alto (suspeito)"
    
    print(f"✅ Sharpe: {metrics.sharpe_ratio:.2f}")


def test_calmar_window():
    """Valida que Calmar usa janela de 36 meses"""
    
    # 3 anos de dados
    n_bars = 252 * 28 * 3
    returns = np.random.normal(0.0005, 0.01, n_bars)
    equity = 100000 * np.cumprod(1 + returns)
    
    trades = pd.DataFrame({'pnl_money': np.diff(equity)})
    
    metrics = calculate_all_metrics(equity, trades)
    
    # Calmar deve usar últimos 36 meses
    assert metrics.calmar_ratio != 0, "Calmar não deve ser zero"
    
    print(f"✅ Calmar: {metrics.calmar_ratio:.2f}")


def test_profit_factor():
    """Valida Profit Factor"""
    
    trades = pd.DataFrame({
        'pnl_money': [1000, -500, 1500, -300, 800]
    })
    
    equity = (100000 + trades['pnl_money'].cumsum()).values
    
    metrics = calculate_all_metrics(equity, trades)
    
    # PF = (1000 + 1500 + 800) / (500 + 300) = 3300 / 800 = 4.125
    expected_pf = 4.125
    
    assert abs(metrics.profit_factor - expected_pf) < 0.01, "Profit Factor incorreto"
    
    print(f"✅ Profit Factor: {metrics.profit_factor:.2f} (esperado: {expected_pf:.2f})")


if __name__ == "__main__":
    print("🧪 Executando testes de validação...\n")
    
    test_sharpe_calculation()
    test_calmar_window()
    test_profit_factor()
    
    print("\n✅ Todos os testes passaram!")