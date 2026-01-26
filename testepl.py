#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Diagnóstico - Verifica logs de trades
Identifica problemas com P&L zerado ou motivos faltando
"""

import os
import re
from datetime import datetime
from collections import defaultdict

def analisar_log_trades(filename):
    """
    Analisa arquivo de log e identifica problemas
    """
    if not os.path.exists(filename):
        print(f"❌ Arquivo não encontrado: {filename}")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 ANÁLISE DO ARQUIVO: {filename}")
    print(f"{'='*80}\n")
    
    estatisticas = {
        'total_linhas': 0,
        'entradas': 0,
        'saidas': 0,
        'pnl_zerado_entrada': 0,
        'pnl_zerado_saida': 0,  # ❌ PROBLEMA!
        'motivo_vazio': 0,
        'lucros': 0,
        'perdas': 0,
        'total_pnl': 0.0
    }
    
    problemas = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    for num_linha, linha in enumerate(linhas, 1):
        # Ignora cabeçalho
        if 'DATA/HORA' in linha or '---' in linha:
            continue
        
        estatisticas['total_linhas'] += 1
        
        try:
            # Extrai campos da linha
            partes = linha.split('|')
            
            if not is_valid_dataframe(partes, min_rows=8):
                problemas.append(f"Linha {num_linha}: Formato inválido")
                continue
            
            timestamp = partes[0].strip()
            tipo = partes[1].strip()
            symbol = partes[2].strip()
            side = partes[3].strip()
            volume_str = partes[4].strip()
            price_str = partes[5].strip()
            pnl_str = partes[6].strip()
            motivo = partes[7].strip() if len(partes) > 7 else ""
            
            # Conta tipo
            if tipo == "ENTRADA":
                estatisticas['entradas'] += 1
            elif tipo == "SAÍDA":
                estatisticas['saidas'] += 1
            
            # Extrai P&L
            match_pnl = re.search(r'P&L:\s*([+-]?\d+\.?\d*)\s*R\$', pnl_str)
            if match_pnl:
                pnl_value = float(match_pnl.group(1))
                
                # 🔴 PROBLEMA: SAÍDA com P&L zerado
                if tipo == "SAÍDA" and abs(pnl_value) < 0.01:
                    estatisticas['pnl_zerado_saida'] += 1
                    problemas.append(
                        f"❌ Linha {num_linha}: SAÍDA com P&L ZERADO!\n"
                        f"   {symbol} {side} | {motivo}"
                    )
                
                # Contabiliza
                if pnl_value > 0:
                    estatisticas['lucros'] += 1
                elif pnl_value < 0:
                    estatisticas['perdas'] += 1
                
                estatisticas['total_pnl'] += pnl_value
            
            # Verifica motivo vazio
            if not motivo or motivo == "Motivo:":
                estatisticas['motivo_vazio'] += 1
                problemas.append(
                    f"⚠️ Linha {num_linha}: Motivo vazio\n"
                    f"   {symbol} {side} {tipo}"
                )
        
        except Exception as e:
            problemas.append(f"Erro ao processar linha {num_linha}: {e}")
    
    # RELATÓRIO
    print("📈 ESTATÍSTICAS:")
    print(f"   Total de operações: {estatisticas['total_linhas']}")
    print(f"   Entradas: {estatisticas['entradas']}")
    print(f"   Saídas: {estatisticas['saidas']}")
    print(f"   Lucros: {estatisticas['lucros']}")
    print(f"   Perdas: {estatisticas['perdas']}")
    print(f"   P&L Total: R${estatisticas['total_pnl']:+,.2f}\n")
    
    # PROBLEMAS ENCONTRADOS
    if estatisticas['pnl_zerado_saida'] > 0:
        print(f"🚨 PROBLEMA CRÍTICO: {estatisticas['pnl_zerado_saida']} SAÍDAS com P&L ZERADO!")
        print("   Isso indica que close_position() não está calculando P&L corretamente\n")
    
    if estatisticas['motivo_vazio'] > 0:
        print(f"⚠️ {estatisticas['motivo_vazio']} operações sem motivo registrado\n")
    
    if problemas:
        print(f"\n{'='*80}")
        print(f"🔍 DETALHES DOS PROBLEMAS ({len(problemas)}):")
        print(f"{'='*80}\n")
        for prob in problemas[:10]:  # Mostra até 10
            print(prob)
            print()
        
        if len(problemas) > 10:
            print(f"... e mais {len(problemas) - 10} problemas")
    else:
        print("✅ Nenhum problema crítico encontrado!")
    
    print(f"\n{'='*80}\n")


def main():
    """
    Analisa todos os arquivos de log de hoje
    """
    hoje = datetime.now().strftime('%Y-%m-%d')
    arquivo_hoje = f"trades_log_{hoje}.txt"
    
    print("🔍 DIAGNÓSTICO DE LOGS DE TRADES")
    print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    # Analisa arquivo de hoje
    analisar_log_trades(arquivo_hoje)
    
    # Lista outros arquivos disponíveis
    print("\n📁 OUTROS LOGS DISPONÍVEIS:")
    logs_encontrados = [f for f in os.listdir('.') if f.startswith('trades_log_') and f.endswith('.txt')]
    
    if logs_encontrados:
        for log in sorted(logs_encontrados):
            size = os.path.getsize(log) / 1024  # KB
            print(f"   • {log} ({size:.1f} KB)")
    else:
        print("   Nenhum outro log encontrado")
    
    print("\n" + "="*80)
    print("💡 SUGESTÕES:")
    print("="*80)
    print("1. Se houver SAÍDAS com P&L zerado, atualize close_position() no bot.py")
    print("2. Verifique se o Telegram mostra valores corretos mas o TXT não")
    print("3. Adicione logs de debug antes de gravar no arquivo")
    print("4. Execute este script após cada sessão de trading")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()