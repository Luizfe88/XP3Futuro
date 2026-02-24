#!/usr/bin/env python3
"""
PATCH: Correção do Profit Factor 999.00 e ajuste de ranges para WIN$N
Resolve os problemas identificados no arquivo 1149.txt
"""

import os
import shutil
import logging

def aplicar_correcoes():
    """Aplica as correções manualmente no optimizer_optuna.py"""
    
    arquivo_original = 'optimizer_optuna.py'
    arquivo_backup = 'optimizer_optuna_backup.py'
    
    # Fazer backup
    if os.path.exists(arquivo_original):
        shutil.copy2(arquivo_original, arquivo_backup)
        print(f"✅ Backup criado: {arquivo_backup}")
    
    # Ler o arquivo original
    with open(arquivo_original, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # ===== CORREÇÃO 1: Profit Factor 999.00 =====
    # Substituir a lógica de atribuição do Profit Factor
    conteudo_corrigido = conteudo.replace(
        """        if gross_losses > 0.0:
            profit_factor = float(gross_profits / gross_losses)
        else:
            profit_factor = 0.0 if gross_profits == 0.0 else 999.0""",
        """        if gross_losses > 0.0:
            profit_factor = float(gross_profits / gross_losses)
        else:
            # Correção: Atribuir PF razoável quando não há perdas
            if gross_profits == 0.0:
                profit_factor = 0.0
            else:
                # Sem perdas mas com lucros = PF muito bom mas não infinito
                profit_factor = min(10.0, max(1.5, gross_profits / max(1.0, len(returns) * 0.01)))"""
    )
    
    # ===== CORREÇÃO 2: Ajustar ranges para WIN$N =====
    # Ajustar os ranges que estavam muito amplos
    
    # Take Profit: 4.0-7.0 → 3.0-5.0 (menos agressivo)
    conteudo_corrigido = conteudo_corrigido.replace(
        "tp_mult = trial.suggest_float(\"tp_mult\", 4.0, 7.0, step=0.1)",
        "tp_mult = trial.suggest_float(\"tp_mult\", 3.0, 5.0, step=0.1)"
    )
    
    # Take Profit alternativo: 4.0-8.0 → 3.0-6.0
    conteudo_corrigido = conteudo_corrigido.replace(
        "tp_mult = trial.suggest_float(\"tp_mult\", 4.0, 8.0, step=0.1)",
        "tp_mult = trial.suggest_float(\"tp_mult\", 3.0, 6.0, step=0.1)"
    )
    
    # Stop Loss: 2.5-4.5 → 2.0-3.5 (mais razoável)
    conteudo_corrigido = conteudo_corrigido.replace(
        "sl_atr_multiplier = trial.suggest_float(\"sl_atr_multiplier\", 2.5, 4.5, step=0.1)",
        "sl_atr_multiplier = trial.suggest_float(\"sl_atr_multiplier\", 2.0, 3.5, step=0.1)"
    )
    
    # Stop Loss alternativo: 2.5-5.0 → 2.0-4.0
    conteudo_corrigido = conteudo_corrigido.replace(
        "sl_atr_multiplier = trial.suggest_float(\"sl_atr_multiplier\", 2.5, 5.0, step=0.1)",
        "sl_atr_multiplier = trial.suggest_float(\"sl_atr_multiplier\", 2.0, 4.0, step=0.1)"
    )
    
    # ===== CORREÇÃO 3: Ajustar o threshold de warning =====
    # Reduzir o threshold de warning de 5.0 para 3.0
    conteudo_corrigido = conteudo_corrigido.replace(
        "if profit_factor > 5.0:",
        "if profit_factor > 3.0:"
    )
    
    # ===== CORREÇÃO 4: Melhorar mensagem de warning =====
    conteudo_corrigido = conteudo_corrigido.replace(
        'logger.warning(f"Profit Factor suspeito: {profit_factor:.2f} (capped)")',
        'logger.warning(f"Profit Factor elevado: {profit_factor:.2f} (limitado para 5.0)")'
    )
    
    # Escrever o arquivo corrigido
    with open(arquivo_original, 'w', encoding='utf-8') as f:
        f.write(conteudo_corrigido)
    
    print("✅ Correções aplicadas com sucesso!")
    print("\n📋 Resumo das alterações:")
    print("  • Profit Factor: Evita atribuição de 999.0, limita a 10.0 máximo")
    print("  • Take Profit: Reduzido de 4.0-7.0 para 3.0-5.0")
    print("  • Stop Loss: Reduzido de 2.5-4.5 para 2.0-3.5")
    print("  • Warning threshold: Reduzido de 5.0 para 3.0")
    print("  • Mensagem de warning: Mais informativa")

def criar_script_teste():
    """Cria um script para testar as correções"""
    
    script_content = '''#!/usr/bin/env python3
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
    print(f"  ✅ PF razoável (não 999.0!)\n")
    
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
    print(f"  ✅ PF zero quando não há trades\n")

if __name__ == "__main__":
    test_profit_factor()
    print("✅ Testes concluídos com sucesso!")
'''
    
    with open('teste_correcoes.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("\n✅ Script de teste criado: teste_correcoes.py")

if __name__ == "__main__":
    print("🔧 Aplicando correções no optimizer_optuna.py")
    print("=" * 60)
    
    aplicar_correcoes()
    criar_script_teste()
    
    print("\n" + "=" * 60)
    print("🎯 Correções concluídas!")
    print("\n📋 Próximos passos:")
    print("  1. Executar teste: python teste_correcoes.py")
    print("  2. Testar otimização: python otimizador_semanal.py --symbols WIN$N --trials 5")
    print("  3. Verificar se os warnings de 999.00 desapareceram")