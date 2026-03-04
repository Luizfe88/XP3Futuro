# watchdog.py - Watchdog robusto com NOVA JANELA VISÍVEL para o bot

import psutil
import subprocess
import sys
import time
import os
import datetime
import platform
import MetaTrader5 as mt5
from pathlib import Path
import psutil
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("watchdog_debug.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==================== CONFIGURAÇÕES ====================
BOT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "botfuturo.py")
LOG_FILE = "xp3_bot.log"               # Arquivo de log do bot
MAX_INACTIVITY_SECONDS = 180           # 3 minutos sem log = suspeito
CHECK_INTERVAL = 30                    # Verifica a cada 30s
MT5_TIMEOUT_SECONDS = 10               # Timeout para testar MT5
# ======================================================



def is_bot_running():
    \"\"\"Verifica se o processo do bot.py está ativo (exclui o watchdog)\"\"\"
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Pasta do watchdog (xp3future)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmd = ' '.join(proc.info['cmdline']).lower()
                # Verifica se é Python, tem bot.py, não é watchdog E está no diretório correto
                if ('python' in proc.info['name'].lower() and 
                    ('bot.py' in cmd or 'botfuturo.py' in cmd) and 
                    'watchdog.py' not in cmd and 'botwatchdog.py' not in cmd and
                    'xp3future' in cmd):  # CRÍTICO: apenas processos do xp3future
                    return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def get_log_last_modified():
    if not os.path.exists(LOG_FILE):
        return None
    return os.path.getmtime(LOG_FILE)

def is_mt5_connected():
    try:
        if mt5.initialize(timeout=MT5_TIMEOUT_SECONDS * 1000):
            connected = mt5.terminal_info() is not None and mt5.account_info() is not None
            mt5.shutdown()
            return connected
    except:
        return False
    return False

def kill_bot(pid):
    try:
        proc = psutil.Process(pid)
        proc.kill()
        print(f"[{datetime.datetime.now()}] ⚡ Bot morto (PID {pid})")
    except:
        pass

def start_bot_in_new_window():
    """Inicia o bot em uma NOVA JANELA DE TERMINAL visível"""
    system = platform.system()
    
    print(f"[{datetime.datetime.now()}] 🚀 Reiniciando bot em NOVA JANELA ({system})...")
    logging.info("Start bot GUI: system=%s", system)
    
    if system == "Windows":
        # Correção para PowerShell e CMD - usa 'start' do shell
        # O título da janela deve ser o primeiro argumento do start
        cmd = f'start "XP3 FUTURE BOT" cmd /k "{sys.executable} {BOT_SCRIPT}"'
        subprocess.Popen(cmd, shell=True)
        logging.info(f"Bot process started via shell command: {cmd}")
        
    elif system == "Linux":
        # Tenta vários terminais comuns (gnome, xfce, kde, etc.)
        terminals = [
            ["gnome-terminal", "--", "python3", BOT_SCRIPT],
            ["konsole", "-e", "python3", BOT_SCRIPT],
            ["xfce4-terminal", "-e", f"python3 {BOT_SCRIPT}"],
            ["xterm", "-e", f"python3 {BOT_SCRIPT}"],
            ["lxterminal", "-e", f"python3 {BOT_SCRIPT}"]
        ]
        for term_cmd in terminals:
            try:
                subprocess.Popen(term_cmd)
                logging.info("Bot process started via terminal: %s", term_cmd[0])
                return  # Sucesso → sai
            except FileNotFoundError:
                continue
        print("⚠️ Nenhum terminal encontrado! Instale gnome-terminal ou similar.")
        
    elif system == "Darwin":  # macOS
        subprocess.Popen([
            "osascript", "-e",
            f'tell app "Terminal" to do script "python3 {BOT_SCRIPT}"'
        ])
        logging.info("Bot process started via osascript Terminal")
    else:
        print("⚠️ Sistema operacional não suportado para nova janela.")
        logging.warning("Unsupported OS for visible GUI start")

def main():
    print(f"[{datetime.datetime.now()}] 🐶 Watchdog iniciado - Monitorando {BOT_SCRIPT}")
    print("   → Ao detectar problema, abrirá NOVA JANELA com o painel do bot")
    
    last_log_time = get_log_last_modified()
    
    while True:
        time.sleep(CHECK_INTERVAL)
        current_time = datetime.datetime.now()
        pid = is_bot_running()
        
        # 1. Bot não está rodando → inicia em nova janela
        if pid is None:
            print(f"[{current_time}] ❌ Bot parado → Abrindo nova janela...")
            start_bot_in_new_window()
            time.sleep(15)  # Dá tempo para o bot iniciar
            last_log_time = get_log_last_modified()
            continue
        
        # 2. Log parado há muito tempo (freeze)
        current_log_time = get_log_last_modified()
        if current_log_time and last_log_time and (current_log_time - last_log_time) > MAX_INACTIVITY_SECONDS:
            print(f"[{current_time}] ⏰ Freeze detectado ({current_log_time - last_log_time:.0f}s sem log) → Reiniciando")
            kill_bot(pid)
            start_bot_in_new_window()
            time.sleep(15)
            last_log_time = get_log_last_modified()
            continue
        
        # 3. MT5 não responde
        if not is_mt5_connected():
            print(f"[{current_time}] 📡 MT5 sem resposta → Reiniciando")
            kill_bot(pid)
            start_bot_in_new_window()
            time.sleep(20)
            last_log_time = get_log_last_modified()
            continue
        
        # Tudo ok
        if current_log_time != last_log_time:
            last_log_time = current_log_time
        
        print(f"[{current_time}] ✅ Bot rodando normalmente (PID {pid})")

if __name__ == "__main__":
    main()
