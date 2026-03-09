import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from hmmlearn.hmm import GaussianHMM
from datetime import datetime

# ==========================================
# 1. FILTRO DE KALMAN (DE PHASE 1)
# ==========================================
class KalmanFilter1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-3):
        self.Q = process_variance 
        self.R = measurement_variance 
        self.x_hat = None
        self.p_hat = 1.0

    def update(self, z):
        if self.x_hat is None:
            self.x_hat = z
            return self.x_hat
        x_pred = self.x_hat
        p_pred = self.p_hat + self.Q
        K = p_pred / (p_pred + self.R)
        self.x_hat = x_pred + K * (z - x_pred)
        self.p_hat = (1 - K) * p_pred
        return self.x_hat

# ==========================================
# 2. TREINAMENTO HMM (PHASE 2)
# ==========================================
def train_and_plot_hmm(df, kalman_col='kalman_price', n_states=3):
    """
    Treina um Hidden Markov Model para identificar Regimes de Mercado.
    """
    print(f"🚀 Treinando HMM com {n_states} estados latentes...")
    
    # 1. Engenharia de Features (Baseada no sinal purificado)
    df['kalman_returns'] = df[kalman_col].pct_change()
    df['kalman_volatility'] = df['kalman_returns'].rolling(window=15).std()
    
    # Prepara os dados removendo NaNs
    feature_cols = ['kalman_returns', 'kalman_volatility']
    data_clean = df.dropna(subset=feature_cols).copy()
    X = data_clean[feature_cols].values
    
    # 2. Construção e Treinamento do Modelo HMM
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # 3. Predição dos Estados Latentes
    data_clean['regime_hmm'] = model.predict(X)
    
    # Adiciona de volta ao dataframe original
    df = df.join(data_clean[['regime_hmm']])
    
    # 4. Visualização dos Regimes
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Identificação Automática de Estados se n_states=3
    if n_states == 3:
        means = model.means_
        # Feature 0: Retorno Médio, Feature 1: Volatilidade Média
        volatilities = means[:, 1]
        returns = means[:, 0]
        
        # O estado de menor volatilidade é o Choppy/Consolidação
        choppy_state = np.argmin(volatilities)
        
        # Dos outros dois, o de maior retorno é o BULL e o de menor retorno é o BEAR
        remaining_states = [i for i in range(3) if i != choppy_state]
        if returns[remaining_states[0]] > returns[remaining_states[1]]:
            bull_state, bear_state = remaining_states[0], remaining_states[1]
        else:
            bull_state, bear_state = remaining_states[1], remaining_states[0]
            
        colors = {bull_state: 'green', bear_state: 'red', choppy_state: 'gray'}
        labels = {bull_state: 'BULL (Tendência Alta)', bear_state: 'BEAR (Tendência Baixa)', choppy_state: 'CHOPPY (Consolidação)'}
    else:
        # Lógica para 2 estados
        state_means = model.means_[:, 1]
        high_vol_state = np.argmax(state_means)
        low_vol_state = np.argmin(state_means)
        colors = {high_vol_state: 'red', low_vol_state: 'green'}
        labels = {high_vol_state: 'Volatilidade/Risco', low_vol_state: 'Estável/Tendência'}
    
    for i in range(n_states):
        idx = df[df['regime_hmm'] == i].index
        if not idx.empty:
            ax.scatter(idx, df.loc[idx, 'close'], c=colors[i], label=labels[i], s=10, alpha=0.4)
        
    ax.plot(df.index, df[kalman_col], color='black', linewidth=1.0, label='Sinal Kalman', alpha=0.7)
    ax.set_title(f"Regimes de Mercado (HMM {n_states} Estados) no WIN$N")
    ax.set_ylabel("Preço")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = f'hmm_regimes_win_{n_states}states.png'
    plt.savefig(filename)
    print(f"📊 Gráfico de regimes salvo como '{filename}'")
    
    return df, model

# ==========================================
# 3. EXECUÇÃO FASE 2
# ==========================================
def run_phase_2(n_states=3): # Alterado para 3 por padrão
    print(f"🎯 Iniciando Fase 2: Identificação de Regimes via HMM ({n_states} estados)")
    
    terminal_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=terminal_path):
        print("❌ Falha ao inicializar MT5")
        return

    symbol = "WIN$N"
    print(f"📡 Extraindo dados para treinamento...")
    
    # Pegamos um período maior (5000 barras M1) para o HMM aprender padrões significativos
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5000)
    if rates is None or len(rates) == 0:
        print(f"❌ Falha ao extrair dados")
        mt5.shutdown()
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Passo 1: Purificação com Kalman (Q=1e-4, R=1e-3 validado na Fase 1)
    kf = KalmanFilter1D(process_variance=1e-4, measurement_variance=1e-3)
    df['kalman_price'] = [kf.update(z) for z in df['close'].values]
    
    # Passo 2: Treinamento HMM
    df, model = train_and_plot_hmm(df)
    
    # Estatísticas dos Regimes
    for i in range(model.n_components):
        state_data = df[df['regime_hmm'] == i]
        print(f"\n--- Estatísticas Regime {i} ---")
        print(f"Ocorrências: {len(state_data)}")
        print(f"Volatilidade Média (Purificada): {state_data['kalman_returns'].std():.6f}")
        print(f"Retorno Médio: {state_data['kalman_returns'].mean():.6f}")

    mt5.shutdown()
    # plt.show() # Desativado para execução em background

if __name__ == "__main__":
    run_phase_2()
