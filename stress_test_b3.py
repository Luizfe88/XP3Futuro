
import pandas as pd
import numpy as np
import logging
import time
import os
import sys

# Adiciona diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validation import validate_position_risk
from hedging import PredictiveHedger, apply_hedge
import config
from utils import calculate_daily_dd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stress_test")

class StressTester:
    """
    Simula o comportamento do XP3 PRO em cenários de crise extrema (Crash 2020, Joesley Day, etc.)
    Foca em Risk Management e Hedging.
    """
    
    def __init__(self):
        self.scenarios = {
            "CRASH_COVID_2020": {
                "description": "Volatilidade extrema e drawdown rápido (Circuit Breakers)",
                "data": self._generate_scenario_data(-0.15, 0.45) # -15% drop, 400% vol
            },
            "JOESLEY_DAY_2017": {
                "description": "Gap de abertura monstro (-10%) e pânico generalizado",
                "data": self._generate_scenario_data(-0.10, 0.35, gap=True)
            },
            "ELEICOES_VOL": {
                "description": "Alta volatilidade intraday sem direção definida (Chop)",
                "data": self._generate_scenario_data(-0.02, 0.25, chop=True)
            }
        }
        self.hedger = PredictiveHedger()
        
    def _generate_scenario_data(self, drop_pct, vol_increase, gap=False, chop=False):
        """Gera dados sintéticos baseados em características reais"""
        # Simples simulação de candles de 1min
        data = []
        price = 100.0
        
        if gap:
            price = price * (1 + drop_pct) # Aplica gap inicial
            
        for i in range(60): # 1 hora de simulação
            vol = 0.02 * (1 + vol_increase) # Base 2% vol diária normalizada
            
            if chop:
                change = np.random.normal(0, vol/5) # Random walk
            else:
                change = np.random.normal(drop_pct/60, vol/10) # Drift negativo constante
                
            price = price * (1 + change)
            
            # Simula VIX disparando se preço cai
            simulated_vix = 20 + (100 - price) * 0.5 
            
            data.append({
                "time": i,
                "price": price,
                "vix": simulated_vix,
                "drawdown": (100 - price) / 100
            })
            
        return data

    def run(self):
        print(f"\n🔥 INICIANDO STRESS TEST XP3 PRO 🔥")
        print("====================================")
        
        results = {}
        
        for name, scenario in self.scenarios.items():
            print(f"\n➡️ Simulando: {name}")
            print(f"📖 {scenario['description']}")
            
            data = scenario['data']
            triggered_hedges = 0
            max_dd = 0
            survived = True
            
            for step in data:
                # Simula estado do bot
                current_price = step['price']
                current_vix = step['vix']
                current_dd = step['drawdown']
                
                max_dd = max(max_dd, current_dd)
                
                # Check 1: Circuit Breaker do Bot (config.MAX_DAILY_DRAWDOWN)
                if current_dd > 0.05: # Assumindo 5% limite
                    print(f"   🛑 Circuit Breaker Simulado ativado! DD: {current_dd:.1%}")
                    survived = False # Parou de operar (bom sinal de segurança)
                    break
                    
                # Check 2: Hedging Trigger
                # Mocking utils.get_vix_br and drawdown logic
                risk_score = self.hedger.calculate_risk_score(current_dd, current_vix, 0, 15, heat=0.8)
                
                should_hedge, reason = self.hedger.should_hedge(risk_score, "Increasing", current_vix)
                
                if should_hedge:
                    triggered_hedges += 1
                    # print(f"   🛡️ Hedge Triggered @ step {step['time']}: {reason}")
            
            results[name] = {
                "max_drawdown": max_dd,
                "hedges_triggered": triggered_hedges,
                "safety_stop": not survived
            }
            
            print(f"   ✅ Resultado: Max DD {max_dd:.1%} | Hedges: {triggered_hedges} | Segurança Ativa: {not survived}")

        print("\n📊 RELATÓRIO FINAL")
        print(pd.DataFrame(results).T)

if __name__ == "__main__":
    tester = StressTester()
    tester.run()
