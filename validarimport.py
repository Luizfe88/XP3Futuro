#!/usr/bin/env python3
"""
Script de Validação - XP3 Trading Bot
Verifica imports ausentes e outros problemas comuns
"""

import re
import sys
from pathlib import Path


def check_imports(filename):
    """Verifica se todas as funções usadas estão importadas"""
    
    print(f"\n{'='*60}")
    print(f"Verificando: {filename}")
    print(f"{'='*60}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filename}")
        return False
    
    # 1. Encontra todos os imports de utils
    import_pattern = r'from utils import \((.*?)\)'
    imports_match = re.search(import_pattern, content, re.DOTALL)
    
    if not imports_match:
        print("⚠️  Nenhum import de utils encontrado")
        return False
    
    imported_functions = set()
    import_text = imports_match.group(1)
    
    # Parse dos imports (remove comentários e espaços)
    for line in import_text.split(','):
        func = line.strip().split('#')[0].strip()
        if func:
            imported_functions.add(func)
    
    print(f"\n✅ Funções importadas de utils ({len(imported_functions)}):")
    for func in sorted(imported_functions):
        print(f"   - {func}")
    
    # 2. Encontra todas as chamadas de funções que parecem vir de utils
    # Padrão: funções que começam com minúscula e não são built-ins
    usage_pattern = r'\b([a-z_][a-z0-9_]*)\s*\('
    used_functions = set(re.findall(usage_pattern, content))
    
    # Lista de built-ins e funções locais conhecidas
    builtins = {
        'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 
        'dict', 'set', 'tuple', 'open', 'max', 'min', 'sum', 'abs', 
        'round', 'sorted', 'enumerate', 'zip', 'map', 'filter', 'all', 
        'any', 'isinstance', 'hasattr', 'getattr', 'setattr', 'type',
        'append', 'extend', 'insert', 'pop', 'remove', 'clear', 'copy',
        'update', 'get', 'items', 'keys', 'values', 'split', 'join',
        'strip', 'replace', 'format', 'strftime', 'isoformat', 'sleep',
        'time', 'datetime', 'timedelta', 'date'
    }
    
    # Remove built-ins
    used_functions -= builtins
    
    # 3. Procura especificamente por is_valid_dataframe
    if 'is_valid_dataframe' in content:
        print(f"\n🔍 Encontrado uso de 'is_valid_dataframe':")
        
        # Conta quantas vezes é usado
        count = content.count('is_valid_dataframe(')
        print(f"   Usado {count} vez(es)")
        
        # Verifica se foi importado
        if 'is_valid_dataframe' in imported_functions:
            print(f"   ✅ Corretamente importado")
        else:
            print(f"   ❌ NÃO IMPORTADO - ERRO CRÍTICO!")
            print(f"\n   Adicione esta linha aos imports:")
            print(f"   is_valid_dataframe,  # ← ADICIONAR AQUI")
            return False
    
    # 4. Verifica outras funções potencialmente ausentes
    potential_utils_functions = {
        'calculate_signal_score', 'safe_copy_rates', 'get_avg_volume',
        'calculate_correlation_matrix', 'detect_market_regime',
        'macro_trend_ok', 'is_power_hour', 'get_time_bucket',
        'send_telegram_trade', 'send_telegram_exit', 'get_telegram_bot',
        'calculate_position_size_atr', 'validate_order_params',
        'analyze_order_book_depth', 'is_spread_acceptable',
        'calculate_dynamic_sl_tp', 'send_order_with_sl_tp',
        'get_current_risk_pct', 'update_adaptive_weights',
        'record_trade_outcome', 'is_symbol_blocked',
        'calculate_sector_exposure_pct', 'get_cached_indicators',
        'calcular_lucro_realizado_txt', 'send_telegram_message',
        'send_daily_performance_report', 'adjust_global_sl_after_pyr',
        'load_loss_streak_data', 'save_loss_streak_data',
        'save_adaptive_weights', 'load_adaptive_weights',
        'update_correlations'
    }
    
    missing_imports = []
    for func in potential_utils_functions:
        if func in content and func not in imported_functions:
            # Verifica se não é uma definição local
            if f"def {func}(" not in content:
                missing_imports.append(func)
    
    if missing_imports:
        print(f"\n⚠️  Possíveis imports ausentes ({len(missing_imports)}):")
        for func in sorted(missing_imports):
            print(f"   - {func}")
    else:
        print(f"\n✅ Todos os imports parecem estar corretos")
    
    return len(missing_imports) == 0


def check_dataframe_validations(filename):
    """Verifica se há verificações incorretas de DataFrame"""
    
    print(f"\n{'='*60}")
    print(f"Verificando padrões de DataFrame em: {filename}")
    print(f"{'='*60}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filename}")
        return False
    
    issues = []
    
    for i, line in enumerate(lines, 1):
        # Padrões problemáticos
        if re.search(r'\bif\s+(df|trades|data|positions):\s*$', line):
            issues.append((i, line.strip(), "Use is_valid_dataframe()"))
        
        elif re.search(r'\bif\s+not\s+(df|trades|data|positions):\s*$', line):
            issues.append((i, line.strip(), "Use not is_valid_dataframe()"))
    
    if issues:
        print(f"\n⚠️  Encontrados {len(issues)} padrões problemáticos:")
        for line_num, code, suggestion in issues[:10]:  # Mostra só os primeiros 10
            print(f"\n   Linha {line_num}:")
            print(f"   ❌ {code}")
            print(f"   💡 {suggestion}")
        
        if len(issues) > 10:
            print(f"\n   ... e mais {len(issues) - 10} ocorrências")
        
        return False
    else:
        print(f"\n✅ Nenhum padrão problemático encontrado")
        return True


def main():
    """Executa todas as verificações"""
    
    print("\n" + "="*60)
    print("🔍 VALIDAÇÃO DE CÓDIGO - XP3 TRADING BOT")
    print("="*60)
    
    files_to_check = ['bot.py', 'utils.py']
    all_ok = True
    
    for filename in files_to_check:
        if not Path(filename).exists():
            print(f"\n⚠️  {filename} não encontrado no diretório atual")
            continue
        
        # Verifica imports
        imports_ok = check_imports(filename)
        
        # Verifica padrões de DataFrame
        dataframe_ok = check_dataframe_validations(filename)
        
        if not imports_ok or not dataframe_ok:
            all_ok = False
    
    # Resultado final
    print(f"\n{'='*60}")
    if all_ok:
        print("✅ VALIDAÇÃO COMPLETA - NENHUM PROBLEMA ENCONTRADO")
    else:
        print("❌ PROBLEMAS ENCONTRADOS - CORRIJA ANTES DE EXECUTAR")
    print(f"{'='*60}\n")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())