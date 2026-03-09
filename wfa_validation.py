import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from hmmlearn.hmm import GaussianHMM
from datetime import datetime

# ==========================================
# 1. COMPONENTES DAS FASES ANTERIORES
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

class BayesianRiskManager:
    def __init__(self, base_win_rate=0.55, base_payout=1.5, kelly_fraction=0.10, tick_value=0.20):
        self.base_win_rate = base_win_rate
        self.base_payout = base_payout
        self.kelly_fraction = kelly_fraction 
        self.tick_value = tick_value

    def calculate_pnl(self, hmm_regime, price_change_points):
        """
        Calcula o PnL de um passo OOS baseado na decisão de Kelly.
        hmm_regime: estado detectado pelo modelo HMM treinado (IS).
        price_change_points: variação do preço no período OOS em pontos.
        """
        # Simplificação para o WFA: 
        # Se Regime 0 (Choppy) -> Não opera (PnL = 0)
        # Se Regime 1 (Trend)  -> Opera proporcional ao Kelly
        
        # Como o HMM pode trocar a ordem das labels, o motor WFA precisa identificar o regime trend.
        # Aqui, assumimos que o regime de maior volatilidade (HMM mean[1]) é o que queremos evitar ou operar.
        # Para este backtest, operaremos apenas se o regime for favorável.
        
        if hmm_regime == 0: # Assumindo 0 como estável/trend p/ simulação
            p = self.base_win_rate
            b = self.base_payout
            q = 1 - p
            kelly_optimal = (p * b - q) / b
            
            if kelly_optimal <= 0: return 0.0
            
            # Pnl simples: pontos ganhos/perdidos * contratos (fixo 1 p/ simplificar backtest de sinal)
            return price_change_points * self.tick_value
        
        return 0.0 # No trade

# ==========================================
# 2. MOTOR DE WALK-FORWARD ANALYSIS
# ==========================================

class WalkForwardValidator:
    def __init__(self, data, train_window_size=2000, test_window_size=500):
        self.data = data
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.oos_results = []
        self.risk_manager = BayesianRiskManager()

    def run(self):
        total_bars = len(self.data)
        current_idx = 0
        
        print(f"🚀 Iniciando WFA em {total_bars} barras...")
        
        while (current_idx + self.train_window_size + self.test_window_size) <= total_bars:
            # 1. Janelas IS e OOS
            is_end = current_idx + self.train_window_size
            df_is = self.data.iloc[current_idx:is_end].copy()
            
            oos_end = is_end + self.test_window_size
            df_oos = self.data.iloc[is_end:oos_end].copy()
            
            # 2. Treino (In-Sample)
            # Engenharia baseada no Kalman purificado
            df_is['returns'] = df_is['kalman'].pct_change()
            df_is['vol'] = df_is['returns'].rolling(15).std()
            is_clean = df_is.dropna(subset=['returns', 'vol'])
            
            model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
            model.fit(is_clean[['returns', 'vol']].values)
            
            # Identifica o regime de 'Tendência' (menor volatilidade média no HMM)
            trend_regime = np.argmin(model.means_[:, 1])
            
            # 3. Execução (Out-of-Sample)
            df_oos['returns'] = df_oos['kalman'].pct_change()
            df_oos['vol'] = df_oos['returns'].rolling(15).std()
            oos_clean = df_oos.dropna(subset=['returns', 'vol'])
            
            if not oos_clean.empty:
                # Predição usando modelo congelado do IS
                oos_regimes = model.predict(oos_clean[['returns', 'vol']].values)
                
                # Cálculo de PnL Barra a Barra no OOS
                period_pnl = 0.0
                for i in range(len(oos_clean)):
                    regime = oos_regimes[i]
                    if regime == trend_regime:
                        # Se estamos no regime de tendência, pegamos a variação do preço
                        points_change = oos_clean.iloc[i]['close'] - oos_clean.iloc[i]['open']
                        period_pnl += points_change * 0.20 # 0.20 R$/ponto
                
                self.oos_results.append({
                    'start_time': df_oos.index[0],
                    'end_time': df_oos.index[-1],
                    'pnl': period_pnl
                })
            
            current_idx += self.test_window_size
            
        self.compile_report()

    def compile_report(self):
        df_res = pd.DataFrame(self.oos_results)
        if df_res.empty:
            print("❌ Nenhum resultado OOS gerado.")
            return

        total_pnl = df_res['pnl'].sum()
        win_rate_periods = len(df_res[df_res['pnl'] > 0]) / len(df_res)
        
        print("\n--- RELATÓRIO FINAL WFA (OUT-OF-SAMPLE) ---")
        print(f"Períodos analisados: {len(df_res)}")
        print(f"Lucro Acumulado (OOS): R$ {total_pnl:.2f}")
        print(f"Taxa de Sucesso (Períodos): {win_rate_periods:.2%}")
        
        # Plot da Curva de Capital OOS
        df_res['equity'] = df_res['pnl'].cumsum()
        plt.figure(figsize=(10, 5))
        plt.plot(df_res['end_time'], df_res['equity'], marker='o', linestyle='-', color='blue')
        plt.title("Curva de Capital Out-of-Sample (WFA) - WIN$N")
        plt.xlabel("Tempo")
        plt.ylabel("PnL Acumulado (R$)")
        plt.grid(True, alpha=0.3)
        plt.savefig('wfa_equity_curve.png')
        print("📊 Curva de capital salva como 'wfa_equity_curve.png'")
        
        if total_pnl > 0 and win_rate_periods > 0.4: # Critério de robustez (ajustado p/ sinal puro)
            print("✅ CONCLUSÃO: Modelo robusto. Pronto para Phase 5.")
        else:
            print("⚠️ CONCLUSÃO: Overfitting ou viés detectado. Revisar parâmetros.")

# ==========================================
# 3. EXECUÇÃO
# ==========================================
def main():
    terminal_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=terminal_path):
        print("❌ Falha MT5")
        return

    symbol = "WIN$N"
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 7000) # Dataset maior p/ WFA
    mt5.shutdown()
    
    if rates is None: return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Pré-processamento Kalman
    kf = KalmanFilter1D()
    df['kalman'] = [kf.update(z) for z in df['close']]
    
    # WFA: Treina em 3000 min (~6 dias), testa em 500 min (~1 dia)
    wfa = WalkForwardValidator(df, train_window_size=3000, test_window_size=500)
    wfa.run()

if __name__ == "__main__":
    main()
