import MetaTrader5 as mt5
import re
from datetime import datetime
import pandas as pd
import os
import json
import logging

# Setup minimal logging to avoid mess
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_check")

# Mock or Import necessary parts from utils if possible, 
# but a standalone script is safer to avoid heavy imports
def detect_broker():
    info = mt5.terminal_info()
    return str(getattr(info, "server", "") or "")

def get_futures_candidates(base_code):
    masks = [f"{base_code}*", f"{base_code}$*", f"{base_code}@*", f"{base_code}N*", f"{base_code}Z*"]
    seen = set()
    names = []
    for m in masks:
        res = mt5.symbols_get(m) or []
        for s in res:
            if s.name not in seen:
                seen.add(s.name)
                names.append(s.name)
    
    today = datetime.now()
    out = []
    for n in names:
        info = mt5.symbol_info(n)
        if not info or not getattr(info, "visible", False):
            continue
            
        # Replicates the logic in utils.py
        broker = detect_broker().lower()
        if "xp" in broker:
            is_contract = bool(re.search(rf"^{base_code}([FGHJKMNQUVXZ]\d{{2}}|[NZ])$", n))
        else:
            is_contract = bool(re.search(rf"^{base_code}[FGHJKMNQUVXZ]\d{{2}}$", n))
            
        if not is_contract:
            continue
            
        exp_raw = getattr(info, "expiration_time", None)
        exp = None
        if isinstance(exp_raw, (int, float)) and exp_raw > 0:
            exp = datetime.fromtimestamp(float(exp_raw))
            
        if exp is None:
            # Fallback parser implemented in utils.py
            m_code = re.search(r"([FGHJKMNQUVXZ])(\d{2})$", n)
            if m_code:
                mm, yy = m_code.groups()
                m_map = {'F':1, 'G':2, 'H':3, 'J':4, 'K':5, 'M':6, 'N':7, 'Q':8, 'U':9, 'V':10, 'X':11, 'Z':12}
                try:
                    exp = datetime(2000 + int(yy), m_map.get(mm, 1), 15)
                except: pass

        if exp and exp <= today:
            continue
            
        candidates.append({"symbol": n, "exp": exp, "volume": getattr(info, "volume", 0)})
    return candidates

def get_contrato_atual(simbolo):
    # Standalone implementation of utils logic
    s = (simbolo or "").upper().strip().replace("$", "").replace("N", "")
    base = "".join([c for c in s if c.isalpha()])
    
    # IND/WSP Hotfix logic
    if base == "IND": base = "WIN"
    elif base == "WSP": base = "WDO"
    
    # Get candidates
    masks = [f"{base}*", f"{base}$*", f"{base}@*", f"{base}N*", f"{base}Z*"]
    names = []
    seen = set()
    for m in masks:
        res = mt5.symbols_get(m) or []
        for s in res:
            if s.name not in seen:
                seen.add(s.name)
                names.append(s.name)
    
    today = datetime.now()
    candidates = []
    for n in names:
        info = mt5.symbol_info(n)
        if not info or not getattr(info, "visible", False): continue
        
        broker = detect_broker().lower()
        if "xp" in broker:
            is_contract = bool(re.search(rf"^{base}([FGHJKMNQUVXZ]\d{{2}}|[NZ])$", n))
        else:
            is_contract = bool(re.search(rf"^{base}[FGHJKMNQUVXZ]\d{{2}}$", n))
        
        if not is_contract: continue
        
        exp_raw = getattr(info, "expiration_time", None)
        exp = None
        if isinstance(exp_raw, (int, float)) and exp_raw > 0:
            exp = datetime.fromtimestamp(float(exp_raw))
        if exp is None:
            m_code = re.search(r"([FGHJKMNQUVXZ])(\d{2})$", n)
            if m_code:
                mm, yy = m_code.groups()
                m_map = {'F':1, 'G':2, 'H':3, 'J':4, 'K':5, 'M':6, 'N':7, 'Q':8, 'U':9, 'V':10, 'X':11, 'Z':12}
                try: exp = datetime(2000 + int(yy), m_map.get(mm, 1), 15)
                except: pass
        if exp and exp <= today: continue
        
        # Try to get better volume
        vol = 0
        try:
            mt5.symbol_select(n, True)
            rates = mt5.copy_rates_from_pos(n, mt5.TIMEFRAME_M15, 0, 100)
            if rates is not None and len(rates) > 0:
                vol = sum([x['tick_volume'] for x in rates[-50:]])
        except: pass
        
        candidates.append({"symbol": n, "exp": exp, "volume": vol})
        
    if not candidates: return None
    
    # Sort: Volume desc, expiration asc
    candidates.sort(key=lambda c: (-c["volume"], (c["exp"] - today).days if c["exp"] else 9999))
    return candidates[0]["symbol"]

def main():
    if not mt5.initialize(path=r"C:\MetaTrader 5 Terminal\terminal64.exe"):
        print("Erro MT5")
        return

    bases = ["WIN", "IND", "WDO", "DOL", "WSP", "CCM", "BGI", "ICF"]
    print(f"{'Base':<8} | {'Ticker $N':<12} | {'Resolvido':<12} | {'Status':<15}")
    print("-" * 60)
    
    for b in bases:
        target = f"{b}$N"
        resolved = get_contrato_atual(target)
        
        status = "OK" if resolved else "ERRO"
        if resolved:
            info = mt5.symbol_info(resolved)
            if not info: status = "Não no MarketWatch"
            else:
                tick = mt5.symbol_info_tick(resolved)
                if not tick or tick.bid == 0: status = "Sem Liquidez"
                
        print(f"{b:<8} | {target:<12} | {str(resolved):<12} | {status:<15}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
