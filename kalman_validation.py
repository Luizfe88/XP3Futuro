import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import sys
from datetime import datetime

# ==========================================
# 1. FILTRO DE KALMAN (PURIFICAÇÃO DE SINAL)
# ==========================================
class KalmanFilter1D:
    """
    Filtro de Kalman unidimensional projetado para extrair o estado oculto (preço verdadeiro)
    do ruído estocástico da microestrutura da B3.
    """
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        # Q: Variância do Processo (Velocidade de mudança do fundamento)
        self.Q = process_variance 
        # R: Variância da Medição (Nível de ruído do spread/book)
        self.R = measurement_variance 
        
        self.x_hat = None # Preço purificado
        self.p_hat = 1.0  # Covariância do erro

    def update(self, z):
        if self.x_hat is None:
            self.x_hat = z
            return self.x_hat

        # Predição
        x_pred = self.x_hat
        p_pred = self.p_hat + self.Q

        # Atualização (Medição e Ganho de Kalman)
        K = p_pred / (p_pred + self.R)
        self.x_hat = x_pred + K * (z - x_pred)
        self.p_hat = (1 - K) * p_pred

        return self.x_hat

# ==========================================
# 2. SIMULAÇÃO DE MONTE CARLO (SOBREVIVÊNCIA)
# ==========================================
def monte_carlo_survival_test(base_win_rate, base_payout_ratio, trades_per_simulation=1000, 
                              num_simulations=1000, initial_capital=10000, 
                              risk_per_trade_fraction=0.02, frictional_cost_per_trade=2.0):
    ruin_count = 0
    final_capitals = []
    ruin_threshold = initial_capital * 0.30 # Ruína declarada aos -70%

    for _ in range(num_simulations):
        capital = initial_capital
        ruined = False
        random_outcomes = np.random.rand(trades_per_simulation)
        
        for outcome in random_outcomes:
            risk_amount = capital * risk_per_trade_fraction
            
            if outcome <= base_win_rate: # WIN
                gross_profit = risk_amount * base_payout_ratio
                capital += (gross_profit - frictional_cost_per_trade)
            else: # LOSS
                gross_loss = risk_amount
                capital -= (gross_loss + frictional_cost_per_trade)
                
            if capital <= ruin_threshold:
                ruined = True
                break
                
        if ruined:
            ruin_count += 1
            final_capitals.append(ruin_threshold)
        else:
            final_capitals.append(capital)

    prob_ruin = (ruin_count / num_simulations) * 100
    return prob_ruin, np.median(final_capitals)

# ==========================================
# 3. EXTRAÇÃO E CALIBRAÇÃO
# ==========================================

def run_phase_1():
    print("🚀 Iniciando Fase 1: Kalman + Monte Carlo")
    
    # --- Passo 1: Extração de Dados ---
    terminal_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=terminal_path):
        print("❌ Falha ao inicializar MT5")
        return

    symbol = "WIN$N"
    print(f"📡 Extraindo dados de {symbol}...")
    
    # Tenta pegar 2000 barras M1
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2000)
    if rates is None or len(rates) == 0:
        print(f"❌ Falha ao extrair dados para {symbol}")
        mt5.shutdown()
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    prices = df['close'].values
    
    # --- Passo 2: Calibração do Kalman ---
    # Testando diferentes combinações de Q e R
    # Q: Process Variance (menor = suave, maior = reativo)
    # R: Measurement Variance (maior = ignora ruído)
    
    configs = [
        (1e-5, 1e-3, "Conservador (Suave)"),
        (1e-4, 1e-3, "Equilibrado"),
        (1e-3, 1e-3, "Reativo (Rápido)"),
        (1e-5, 1e-2, "Anti-Ruído (Lento)")
    ]
    
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, prices, label="Preço Real (WIN$N)", color='gray', alpha=0.5)
    
    for q, r, label in configs:
        kf = KalmanFilter1D(process_variance=q, measurement_variance=r)
        purified = [kf.update(z) for z in prices]
        plt.plot(df.index, purified, label=f"Kalman {label} (Q={q}, R={r})")
    
    # Adiciona uma EMA para comparação (benchmark)
    ema20 = df['close'].ewm(span=20).mean()
    plt.plot(df.index, ema20, label="EMA 20 (Benchmark)", linestyle='--', color='black')
    
    plt.title(f"Filtragem de sinal WIN$N - Kalman vs Real vs EMA")
    plt.legend()
    plt.grid(True)
    
    # Salva o gráfico
    plot_filename = "kalman_comparison.png"
    plt.savefig(plot_filename)
    print(f"📊 Gráfico de calibração salvo como '{plot_filename}'")
    
    # --- Passo 3: Monte Carlo Survival ---
    print("\n🎲 Iniciando Teste de Sobrevivência (Monte Carlo)...")
    
    # Métricas alvo informadas ou estimadas para o sistema purificado
    target_win_rate = 0.55 # 55%
    target_payout = 1.5    # 1.5:1
    slippage = 2.0         # R$ 2.00 por contrato (custo total)
    
    prob_ruin, median_cap = monte_carlo_survival_test(
        base_win_rate=target_win_rate,
        base_payout_ratio=target_payout,
        frictional_cost_per_trade=slippage
    )
    
    print("-" * 40)
    print(f"Win Rate Alvo: {target_win_rate*100}%")
    print(f"Payout Alvo: {target_payout}")
    print(f"Custo Friccional: R$ {slippage}")
    print(f"Probabilidade de Ruína: {prob_ruin:.2f}%")
    print(f"Mediana Capital Final: R$ {median_cap:.2f}")
    print("-" * 40)
    
    if prob_ruin < 1.0:
        print("✅ Sobrevivência teórica VALIDADA (< 1% risco de ruína)")
    else:
        print("⚠️ Alerta: Risco de ruína acima do limite aceitável!")

    mt5.shutdown()
    # plt.show()

if __name__ == "__main__":
    run_phase_1()
