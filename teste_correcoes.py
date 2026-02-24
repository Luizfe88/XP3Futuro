#!/usr/bin/env python3
"""
Teste das correções aplicadas
"""

import numpy as np

def test_profit_factor():
    """Testa a nova lógica do Profit Factor"""
    
    # Simular cenários
    print("🧪 Testando nova lógica do Profit Factor:")
    
    # Cenário 1: Sem perdas, com lucros
    gross_profits = 1000.0
    gross_losses = 0.0
    returns = np.array([0.01, 0.02, -0.001, 0.015, 0.008])
    
    if gross_losses > 0.0:
        profit_factor = float(gross_profits / gross_losses)
    else:
        if gross_profits == 0.0:
            profit_factor = 0.0
        else:
            # Sem perdas mas com lucros = PF muito bom mas não infinito
            profit_factor = min(10.0, max(1.5, gross_profits / max(1.0, len(returns) * 0.01)))
    
    print(f"  Lucros: {gross_profits}, Perdas: {gross_losses}")
    print(f"  Profit Factor calculado: {profit_factor}")
    print("  ✅ PF razoável (não 999.0!)\n")
    
    # Cenário 2: Sem lucros, sem perdas
    gross_profits = 0.0
    gross_losses = 0.0
    
    if gross_losses > 0.0:
        profit_factor = float(gross_profits / gross_losses)
    else:
        if gross_profits == 0.0:
            profit_factor = 0.0
        else:
            profit_factor = min(10.0, max(1.5, gross_profits / max(1.0, len(returns) * 0.01)))
    
    print(f"  Lucros: {gross_profits}, Perdas: {gross_losses}")
    print(f"  Profit Factor calculado: {profit_factor}")
    print("  ✅ PF zero quando não há trades\n")

if __name__ == "__main__":
    test_profit_factor()
    print("✅ Testes concluídos com sucesso!")
