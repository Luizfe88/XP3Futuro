#!/usr/bin/env python3
"""
FIX COMPLETO: Limites + Dados MT5
Resolve AMBOS os problemas de uma vez

Execute: python fixcompleto.py [A|B|C]
"""

import os
import sys
import subprocess

def executar_comando(cmd, descricao):
    """Executa comando e mostra progresso"""
    print(f"⏳ {descricao}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {descricao} - OK")
            return True
        else:
            print(f"❌ {descricao} - ERRO")
            if result.stderr:
                print(f"   {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ {descricao} - ERRO: {e}")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           FIX COMPLETO - OTIMIZADOR FUTUROS                      ║
    ║                                                                  ║
    ║  Aplica AMBAS as correções:                                     ║
    ║  1. Ajuste de limites (Solução A/B/C)                           ║
    ║  2. Fix de dados MT5 (1248 barras → 3000+)                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    solucao = None
    if len(sys.argv) >= 2 and sys.argv[1]:
        solucao = sys.argv[1].strip().upper()
    else:
        solucao = os.getenv("XP3_FIX_SOLUCAO", "").strip().upper()
    if solucao not in ['A', 'B', 'C']:
        print("USO: python fixcompleto.py [A|B|C]")
        print()
        print("SOLUÇÕES DISPONÍVEIS:")
        print("  A - SOBREVIVÊNCIA (DD 70%, PF 1.05) - Aceita quase tudo")
        print("  B - MODERADO (DD 55%, PF 1.15, Stops Largos) - Recomendado ⭐")
        print("  C - CONSERVADOR (DD 40%, PF 1.30, M30) - Preservar capital")
        print()
        try:
            solucao = input("Digite A, B ou C e pressione Enter: ").strip().upper()
        except Exception:
            solucao = ""
    
    if solucao not in ['A', 'B', 'C']:
        print(f"❌ ERRO: Solução '{solucao}' inválida!")
        print("   Use: A, B ou C")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print(f"  APLICANDO FIX COMPLETO - SOLUÇÃO {solucao}")
    print("=" * 70)
    print()
    
    etapas_ok = []
    
    # ETAPA 1: Diagnóstico inicial
    print("📊 ETAPA 1: Diagnóstico inicial")
    print("-" * 70)
    
    arquivos_necessarios = [
        'optimizer_optuna.py',
        'otimizador_semanal.py'
    ]
    
    arquivos_ok = True
    for arq in arquivos_necessarios:
        # Tentar encontrar em múltiplos locais
        encontrado = False
        for caminho in [arq, f'/mnt/user-data/uploads/{arq}', f'../{arq}']:
            if os.path.exists(caminho):
                print(f"  ✅ {arq} encontrado em {caminho}")
                encontrado = True
                break
        
        if not encontrado:
            print(f"  ❌ {arq} NÃO encontrado!")
            arquivos_ok = False
    
    if not arquivos_ok:
        print()
        print("❌ Arquivos necessários não encontrados!")
        print("   Coloque optimizer_optuna.py e otimizador_semanal.py no diretório atual.")
        sys.exit(1)
    
    etapas_ok.append(True)
    print()
    
    # ETAPA 2: Ativar série contínua
    print("🔧 ETAPA 2: Ativar série contínua MT5")
    print("-" * 70)
    
    os.environ['XP3_FORCE_CONTINUOUS'] = '1'
    print("  ✅ XP3_FORCE_CONTINUOUS=1 (exportado)")
    
    # Salvar em .bashrc ou arquivo de ambiente
    try:
        with open(os.path.expanduser('~/.bashrc'), 'a') as f:
            f.write('\n# Otimizador Futuros - Série Contínua MT5\n')
            f.write('export XP3_FORCE_CONTINUOUS=1\n')
        print("  ✅ Adicionado ao ~/.bashrc (permanente)")
    except:
        print("  ⚠️  Não foi possível adicionar ao ~/.bashrc (não crítico)")
    
    etapas_ok.append(True)
    print()
    
    # ETAPA 3: Aplicar solução de limites
    print(f"🎯 ETAPA 3: Aplicar Solução {solucao} (Limites)")
    print("-" * 70)
    
    if os.path.exists('aplicar_solucao.py'):
        cmd = f'python aplicar_solucao.py {solucao}'
        if executar_comando(cmd, f"Aplicando Solução {solucao}"):
            etapas_ok.append(True)
        else:
            print("  ⚠️  aplicar_solucao.py falhou, mas continuando...")
            etapas_ok.append(False)
    else:
        print("  ⚠️  aplicar_solucao.py não encontrado (pule se já aplicou)")
        etapas_ok.append(False)
    
    print()
    
    # ETAPA 4: Aplicar patch MT5
    print("📡 ETAPA 4: Patch MT5 (copy_rates_range)")
    print("-" * 70)
    
    if os.path.exists('patch_mt5_range.py'):
        cmd = 'python patch_mt5_range.py'
        if executar_comando(cmd, "Aplicando patch MT5"):
            etapas_ok.append(True)
        else:
            print("  ⚠️  patch_mt5_range.py falhou, mas continuando...")
            etapas_ok.append(False)
    else:
        print("  ⚠️  patch_mt5_range.py não encontrado (pule se já aplicou)")
        etapas_ok.append(False)
    
    print()
    
    # ETAPA 5: Copiar arquivos modificados
    print("📁 ETAPA 5: Copiar arquivos modificados")
    print("-" * 70)
    
    arquivos_para_copiar = [
        (f'optimizer_optuna_SOLUCAO_{solucao}.py', 'optimizer_optuna.py'),
        ('otimizador_semanal_PATCHED.py', 'otimizador_semanal.py')
    ]
    
    copia_ok = True
    for origem, destino in arquivos_para_copiar:
        if os.path.exists(origem):
            try:
                import shutil
                shutil.copy2(origem, destino)
                print(f"  ✅ {origem} → {destino}")
            except Exception as e:
                print(f"  ❌ Erro ao copiar {origem}: {e}")
                copia_ok = False
        else:
            print(f"  ⚠️  {origem} não encontrado")
    
    etapas_ok.append(copia_ok)
    print()
    
    # ETAPA 6: Validação final
    print("✅ ETAPA 6: Validação final")
    print("-" * 70)
    
    validacoes = {
        "XP3_FORCE_CONTINUOUS": os.getenv('XP3_FORCE_CONTINUOUS') == '1',
        "optimizer_optuna.py": os.path.exists('optimizer_optuna.py'),
        "otimizador_semanal.py": os.path.exists('otimizador_semanal.py'),
    }
    
    todas_ok = all(validacoes.values())
    
    for nome, status in validacoes.items():
        simbolo = "✅" if status else "❌"
        print(f"  {simbolo} {nome}")
    
    etapas_ok.append(todas_ok)
    print()
    
    # RESUMO FINAL
    print("=" * 70)
    if all(etapas_ok):
        print("✅ FIX COMPLETO APLICADO COM SUCESSO!")
    else:
        print("⚠️  FIX APLICADO COM ALGUNS AVISOS")
    print("=" * 70)
    print()
    
    print("📊 MODIFICAÇÕES APLICADAS:")
    print()
    
    if solucao == 'A':
        print("  SOLUÇÃO A (Sobrevivência):")
        print("    • Max DD: 70% (limite efetivo: 84%)")
        print("    • Min PF: 1.05 (limite efetivo: 0.84)")
        print("    • Min WR: 15% (limite efetivo: 13.5%)")
        print("    • Expectativa: 5-10 sistemas aprovados")
    elif solucao == 'B':
        print("  SOLUÇÃO B (Moderado): ⭐ RECOMENDADO")
        print("    • Max DD: 55% (limite efetivo: 66%)")
        print("    • Min PF: 1.15 (limite efetivo: 0.92)")
        print("    • Min WR: 18% (limite efetivo: 16.2%)")
        print("    • Stops: 2.5-5x ATR (crítico!)")
        print("    • Expectativa: 3-6 sistemas aprovados")
    elif solucao == 'C':
        print("  SOLUÇÃO C (Conservador):")
        print("    • Max DD: 40% (limite efetivo: 48%)")
        print("    • Min PF: 1.30 (limite efetivo: 1.04)")
        print("    • Min WR: 22% (limite efetivo: 19.8%)")
        print("    • Timeframe: M30 (em vez de M15)")
        print("    • Expectativa: 1-3 sistemas aprovados")
    
    print()
    print("  DADOS MT5:")
    print("    • Série contínua ativada (XP3_FORCE_CONTINUOUS=1)")
    print("    • copy_rates_range (sem limite 1248 barras)")
    print("    • Expectativa: 3000-5000+ barras disponíveis")
    print()
    
    print("🚀 PRÓXIMOS PASSOS:")
    print()
    print("  1. TESTE RÁPIDO (recomendado):")
    print("     python diagnostico_barras.py")
    print()
    print("  2. EXECUTAR OTIMIZAÇÃO:")
    print("     python otimizador_semanal.py --symbols WIN$N --trials 30")
    print()
    print("  3. SE OK, OTIMIZAÇÃO COMPLETA:")
    print("     python otimizador_semanal.py --trials 100")
    print()
    
    print("=" * 70)
    print("📝 DOCUMENTAÇÃO:")
    print()
    print("  • DECISAO_RAPIDA.md - Guia de decisão")
    print("  • SOLUCAO_DADOS_MT5.md - Detalhes técnicos MT5")
    print("  • PROMPT_ANALISTA_SOLUCOES_PRONTAS.md - Completo")
    print()
    print("=" * 70)
    
    if all(etapas_ok):
        sys.exit(0)
    else:
        print()
        print("⚠️  Algumas etapas falharam. Revise manualmente se necessário.")
        sys.exit(1)

if __name__ == "__main__":
    main()
