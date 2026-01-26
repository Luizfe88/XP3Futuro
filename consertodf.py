#!/usr/bin/env python3
"""
Script de Correção Automática - XP3 Trading Bot
Corrige automaticamente os padrões problemáticos de verificação de DataFrame

Uso:
    python fix_dataframe_checks.py --dry-run  # Mostra o que seria alterado
    python fix_dataframe_checks.py --apply    # Aplica as correções
    python fix_dataframe_checks.py --backup   # Cria backup antes de aplicar
"""

import re
import os
import shutil
from datetime import datetime
from pathlib import Path
import argparse


class DataFrameFixer:
    """Corrige padrões problemáticos de verificação de DataFrame"""
    
    def __init__(self, dry_run=True, create_backup=True):
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.changes = []
        
        # Padrões de correção (regex, substituição, descrição)
        self.patterns = [
            # Padrão 1: if df:
            (
                r'(\s+)if\s+(df|trades|data|rates|positions):\s*$',
                r'\1if is_valid_dataframe(\2):',
                'if <var>: → if is_valid_dataframe(<var>):'
            ),
            
            # Padrão 2: if not df:
            (
                r'(\s+)if\s+not\s+(df|trades|data|rates|positions):\s*$',
                r'\1if not is_valid_dataframe(\2):',
                'if not <var>: → if not is_valid_dataframe(<var>):'
            ),
            
            # Padrão 3: if df is None or df.empty:
            (
                r'(\s+)if\s+(\w+)\s+is\s+None\s+or\s+\2\.empty:\s*$',
                r'\1if not is_valid_dataframe(\2):',
                'if <var> is None or <var>.empty: → if not is_valid_dataframe(<var>):'
            ),
            
            # Padrão 4: if df.empty:
            (
                r'(\s+)if\s+(\w+)\.empty:\s*$',
                r'\1if not is_valid_dataframe(\2):',
                'if <var>.empty: → if not is_valid_dataframe(<var>):'
            ),
            
            # Padrão 5: if len(df) < N:
            (
                r'(\s+)if\s+len\((\w+)\)\s*<\s*(\d+):\s*$',
                r'\1if not is_valid_dataframe(\2, min_rows=\3):',
                'if len(<var>) < N: → if not is_valid_dataframe(<var>, min_rows=N):'
            ),
            
            # Padrão 6: if df and len(df) > 0:
            (
                r'(\s+)if\s+(\w+)\s+and\s+len\(\2\)\s*>\s*0:\s*$',
                r'\1if is_valid_dataframe(\2):',
                'if <var> and len(<var>) > 0: → if is_valid_dataframe(<var>):'
            ),
        ]
    
    def create_backup_file(self, filepath):
        """Cria backup do arquivo original"""
        if not self.create_backup:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
        shutil.copy2(filepath, backup_path)
        print(f"📦 Backup criado: {backup_path}")
    
    def fix_file(self, filepath):
        """Aplica correções em um arquivo"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"❌ Arquivo não encontrado: {filepath}")
            return False
        
        print(f"\n📄 Processando: {filepath}")
        
        # Lê o arquivo
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"❌ Erro ao ler arquivo: {e}")
            return False
        
        # Aplica correções linha por linha
        modified = False
        new_lines = []
        file_changes = []
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            
            for pattern, replacement, description in self.patterns:
                if re.match(pattern, line):
                    line = re.sub(pattern, replacement, line)
                    
                    if line != original_line:
                        modified = True
                        change = {
                            'file': str(filepath),
                            'line': line_num,
                            'original': original_line.strip(),
                            'fixed': line.strip(),
                            'description': description
                        }
                        file_changes.append(change)
                        self.changes.append(change)
            
            new_lines.append(line)
        
        # Mostra mudanças
        if file_changes:
            print(f"\n  🔧 {len(file_changes)} correções encontradas:")
            for change in file_changes:
                print(f"    Linha {change['line']}:")
                print(f"      ❌ {change['original']}")
                print(f"      ✅ {change['fixed']}")
        else:
            print("  ✅ Nenhuma correção necessária")
        
        # Aplica mudanças se não for dry-run
        if modified and not self.dry_run:
            self.create_backup_file(filepath)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                print(f"  ✅ Arquivo salvo com correções")
            except Exception as e:
                print(f"  ❌ Erro ao salvar: {e}")
                return False
        
        return modified
    
    def add_helper_function(self, filepath):
        """Adiciona a função is_valid_dataframe() no início do arquivo"""
        filepath = Path(filepath)
        
        helper_code = '''
def is_valid_dataframe(df, min_rows: int = 1) -> bool:
    """
    Valida DataFrame de forma segura.
    
    Args:
        df: Objeto a validar (pode ser DataFrame, lista, None, etc)
        min_rows: Número mínimo de linhas (padrão: 1)
    
    Returns:
        True se válido, False caso contrário
    """
    if df is None:
        return False
    
    if isinstance(df, pd.DataFrame):
        return not df.empty and len(df) >= min_rows
    
    if isinstance(df, (list, tuple)):
        return len(df) >= min_rows
    
    return False

'''
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verifica se já existe
            if 'def is_valid_dataframe' in content:
                print(f"  ℹ️  Helper function já existe em {filepath}")
                return False
            
            # Encontra local para inserir (após imports)
            lines = content.split('\n')
            insert_line = 0
            
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_line = i + 1
            
            # Insere helper function
            lines.insert(insert_line, helper_code)
            
            if not self.dry_run:
                self.create_backup_file(filepath)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                print(f"  ✅ Helper function adicionada em {filepath}")
            else:
                print(f"  ℹ️  Helper function seria adicionada em {filepath}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Erro ao adicionar helper: {e}")
            return False
    
    def scan_project(self, directory='.'):
        """Escaneia todos os arquivos Python no projeto"""
        python_files = list(Path(directory).glob('*.py'))
        
        print(f"🔍 Encontrados {len(python_files)} arquivos Python")
        
        # Primeiro adiciona helper function
        if 'utils.py' in [f.name for f in python_files]:
            print("\n📝 Adicionando helper function em utils.py...")
            self.add_helper_function('utils.py')
        
        # Depois processa cada arquivo
        print("\n🔧 Processando arquivos...")
        
        for filepath in python_files:
            self.fix_file(filepath)
        
        return len(self.changes)
    
    def print_summary(self):
        """Imprime resumo das mudanças"""
        print("\n" + "="*70)
        print("📊 RESUMO DAS MUDANÇAS")
        print("="*70)
        
        if not self.changes:
            print("✅ Nenhuma correção necessária!")
            return
        
        print(f"\n🔧 Total de correções: {len(self.changes)}")
        
        # Agrupa por arquivo
        by_file = {}
        for change in self.changes:
            file = change['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(change)
        
        print(f"📁 Arquivos afetados: {len(by_file)}")
        
        for file, changes in by_file.items():
            print(f"\n  📄 {file}: {len(changes)} correções")
        
        if self.dry_run:
            print("\n⚠️  MODO DRY-RUN: Nenhuma alteração foi aplicada")
            print("   Execute com --apply para aplicar as correções")
        else:
            print("\n✅ Correções aplicadas com sucesso!")
            print("   Backups salvos em ./backups/")


def main():
    parser = argparse.ArgumentParser(
        description='Corrige verificações problemáticas de DataFrame'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mostra o que seria alterado sem aplicar mudanças'
    )
    
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Aplica as correções nos arquivos'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Não cria backup dos arquivos originais'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Processa apenas um arquivo específico'
    )
    
    args = parser.parse_args()
    
    # Determina modo de execução
    dry_run = not args.apply
    create_backup = not args.no_backup
    
    if dry_run:
        print("🔍 Modo DRY-RUN ativado (nenhuma alteração será feita)")
    else:
        print("⚠️  Modo APLICAR ativado (arquivos serão modificados)")
        
        if create_backup:
            print("📦 Backups serão criados")
        else:
            print("⚠️  Backups DESATIVADOS!")
    
    print()
    
    # Cria instância do fixer
    fixer = DataFrameFixer(dry_run=dry_run, create_backup=create_backup)
    
    # Processa arquivo(s)
    if args.file:
        fixer.fix_file(args.file)
        fixer.add_helper_function('utils.py')
    else:
        fixer.scan_project()
    
    # Mostra resumo
    fixer.print_summary()


if __name__ == '__main__':
    main()