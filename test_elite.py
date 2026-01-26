
import json
import csv
import os
import shutil
from datetime import datetime
from config import get_elite_settings, ELITE_ASSETS
import logging

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")

def test_elite_logic():
    print("Testing Elite Logic & Config Integration...")
    
    # 1. Create a dummy elite_params.json
    dummy_data = {
        "TEST3": {"rsi_low": 30, "rsi_high": 70, "win_rate": 0.60, "calmar": 2.5}
    }
    
    with open("elite_params.json", 'w') as f:
        json.dump(dummy_data, f)
        
    # 2. Test get_elite_settings
    print("\n--- Testing config.get_elite_settings() ---")
    loaded_data = get_elite_settings()
    
    if "TEST3" in loaded_data:
        print("✅ Config loaded Elite params successfully.")
    else:
        print(f"❌ Config failed to load Elite params. Loaded: {loaded_data}")

    if "TEST3" in ELITE_ASSETS:
         print("✅ ELITE_ASSETS global variable updated.")
    else:
         print(f"❌ ELITE_ASSETS global variable NOT updated.")

    # 3. Simulate Reporter Logic (from otimizador_semanal.py)
    print("\n--- Testing Atomic Save & CSV Report ---")
    
    elite_assets = {"VALE3": {"param": 1}}
    rejected_assets = [{"symbol": "PETR4", "status": "PERFORMANCE", "reason": "WR < 55%", "timestamp": datetime.now().isoformat()}]
    
    # Atomic Save Simulation
    try:
        temp = "elite_params.json.tmp"
        final = "elite_params.json"
        with open(temp, 'w') as f:
            json.dump(elite_assets, f)
        if os.path.exists(final): os.remove(final)
        os.rename(temp, final)
        print("✅ Atomic save simulation success.")
    except Exception as e:
        print(f"❌ Atomic save failed: {e}")

    # CSV Report Simulation
    try:
        with open("rejected_assets_report.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["symbol", "status", "reason", "timestamp"])
            writer.writeheader()
            writer.writerows(rejected_assets)
        print("✅ CSV Report save success.")
    except Exception as e:
        print(f"❌ CSV save failed: {e}")

    # Verify atomic save result
    with open("elite_params.json", 'r') as f:
        saved = json.load(f)
        if "VALE3" in saved:
            print("✅ elite_params.json content verified (Overwritten by simulation).")
        else:
             print("❌ elite_params.json content mismatch.")

if __name__ == "__main__":
    test_elite_logic()
    # Cleanup
    if os.path.exists("elite_params.json"): os.remove("elite_params.json")
    if os.path.exists("rejected_assets_report.csv"): os.remove("rejected_assets_report.csv")
