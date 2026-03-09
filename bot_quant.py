import time
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from execution_engine import ExecutionEngine
from hmm_validation import KalmanFilter1D, train_and_plot_hmm
from risk_validation import BayesianRiskManager

# Configuração de Logging Centralizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/bot_quant_production.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantBot")

class WINQuantBot:
    def __init__(self, symbol="WIN$N", capital_base=10000.0):
        self.symbol = symbol
        self.capital = capital_base
        self.engine = ExecutionEngine(symbol=symbol)
        self.kf = KalmanFilter1D(process_variance=1e-4, measurement_variance=1e-3)
        self.risk_manager = BayesianRiskManager(base_win_rate=0.55, base_payout=1.5, kelly_fraction=0.1)
        
        self.hmm_model = None
        self.last_bar_time = None
        self.is_running = False

    def startup(self):
        """Inicialização e Treinamento Inicial do HMM."""
        if not self.engine.connect():
            return False
        
        logger.info("[START] Treinando modelo HMM inicial...")
        # Pegamos 3000 barras para o treino inicial (Walk-Forward comprovou robustez nessa janela)
        df_init = self.engine.get_latest_m1_data(count=3000)
        if df_init is None:
            logger.error("[FAIL] Falha ao obter dados iniciais.")
            return False
        
        # Prep Kalman
        df_init['kalman'] = [self.kf.update(z) for z in df_init['close']]
        
        # Treino HMM (2 estados conforme validado na Fase 2)
        try:
            from hmmlearn.hmm import GaussianHMM
            # Engenharia de Features local
            df_init['returns'] = df_init['kalman'].pct_change()
            df_init['vol'] = df_init['returns'].rolling(15).std()
            clean = df_init.dropna().copy()
            X = clean[['returns', 'vol']].values.copy()
            X += np.random.normal(0, 1e-9, X.shape)
            
            self.hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
            self.hmm_model.fit(X)
            logger.info("[OK] Cérebro Quant instanciado e treinado.")
            return True
        except Exception as e:
            logger.error(f"[FAIL] Erro ao treinar HMM: {e}")
            return False

    def run(self):
        """Loop principal de execução em tempo real."""
        self.is_running = True
        logger.info(f"[RUN] Bot em Shadow Trading na conta Demo... Ativo: {self.symbol}")
        
        try:
            while self.is_running:
                now = datetime.now()
                if self.last_bar_time is None or now.minute != self.last_bar_time.minute:
                    self.execute_logic()
                    self.last_bar_time = now
                
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def execute_logic(self):
        """Pipeline do sinal: Kalman -> HMM -> Bayes -> Execution."""
        # 1. Fetch data
        df = self.engine.get_latest_m1_data(count=50)
        if df is None: return

        # 2. Update Kalman Signal
        current_price = df['close'].iloc[-1]
        kalman_signal = self.kf.update(current_price)
        
        # 3. Predict HMM Regime and Confidence
        df['kalman'] = [self.kf.update(z) for z in df['close']] 
        df['returns'] = df['kalman'].pct_change()
        df['vol'] = df['returns'].rolling(15).std()
        
        latest_feat = df[['returns', 'vol']].dropna().iloc[-1:].values.copy()
        if len(latest_feat) == 0: return
        
        # Adiciona jitter para estabilidade
        latest_feat_jitter = latest_feat + np.random.normal(0, 1e-9, latest_feat.shape)
        
        regime = self.hmm_model.predict(latest_feat_jitter)[0]
        probs = self.hmm_model.predict_proba(latest_feat_jitter)[0]
        confidence = probs[regime]
        
        # 4. Bayesian Risk Management
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                      abs(df['low'] - df['close'].shift(1))))
        atr = df['tr'].rolling(14).mean().iloc[-1]
        
        contracts, risk_pct, debug_info = self.risk_manager.calculate_position_size(
            total_capital=self.capital, 
            hmm_regime=regime, 
            atr_points=atr,
            confidence=confidence
        )
        
        # Determinar Viés (Direção)
        bias = "COMPRA" if current_price > kalman_signal else "VENDA"
        if contracts == 0:
            bias = "AGUARDAR"
            
        logger.info(f"{self.symbol: <6} | {bias: <8} | Sinal: {current_price:.0f} | Kalman: {kalman_signal:.2f} | Regime: {regime} | Lote: {contracts} | {debug_info}")

        # 5. Execution (Shadow Trading)
        if contracts > 0:
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                order_type = mt5.ORDER_TYPE_BUY if bias == "COMPRA" else mt5.ORDER_TYPE_SELL
                self.engine.execute_market_order(order_type, contracts, signal_price=current_price)
        else:
            # Se Regime 0 (Choppy), poderíamos fechar posições existentes para proteger capital
            self.engine.close_all_positions()

    def stop(self):
        self.is_running = False
        self.engine.close_all_positions()
        self.engine.shutdown()
        logger.info("[STOP] Bot encerrado.")

if __name__ == "__main__":
    bot = WINQuantBot()
    if bot.startup():
        bot.run()
