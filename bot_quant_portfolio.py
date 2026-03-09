import time
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import threading
from datetime import datetime
from execution_engine import ExecutionEngine
from risk_validation import BayesianRiskManager
from hmm_validation import KalmanFilter1D, train_and_plot_hmm
import json
import os

# ==========================================
# 1. CONFIGURAÇÃO DE LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/portfolio_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PortfolioBot")

# ==========================================
# 2. DEFINIÇÕES DO PORTFÓLIO
# ==========================================
PORTFOLIO_CONFIG = {
    "WDO$N": {
        "allocation": 0.30, 
        "tick_value": 5.0, 
        "base_win_rate": 0.55, 
        "base_payout": 1.5,
        "n_states": 2,
        "kelly_fraction": 0.10
    },
    "WSP$N": {
        "allocation": 0.30, 
        "tick_value": 25.0, 
        "base_win_rate": 0.52, 
        "base_payout": 2.0,
        "n_states": 3,
        "kelly_fraction": 0.15
    },
    "WIN$N": {
        "allocation": 0.20, 
        "tick_value": 0.20, 
        "base_win_rate": 0.54, 
        "base_payout": 1.5,
        "n_states": 2,
        "kelly_fraction": 0.08  # Tensão/Ruído alto no WIN, Kelly bem reduzido
    },
    "DI1$N": {
        "allocation": 0.20, 
        "tick_value": 1.0, 
        "base_win_rate": 0.60, 
        "base_payout": 1.2,
        "n_states": 2,
        "kelly_fraction": 0.05
    }
}

class AssetWorker:
    """Instância individual para cada ativo do portfólio."""
    def __init__(self, symbol, config, capital_total):
        self.symbol = symbol
        self.allocation = config['allocation']
        self.capital_total = capital_total
        self.engine = ExecutionEngine(symbol=symbol, magic_number=999000 + list(PORTFOLIO_CONFIG.keys()).index(symbol))
        
        kalman_q = 1e-4
        kalman_r = 1e-3
        base_win_rate = config['base_win_rate']
        base_payout = config['base_payout']
        
        if os.path.exists("calibrated_assets.json"):
            try:
                with open("calibrated_assets.json", "r") as f:
                    calib_data = json.load(f)
                    if symbol in calib_data:
                        kalman_q = calib_data[symbol].get("kalman_q", 1e-4)
                        kalman_r = calib_data[symbol].get("kalman_r", 1e-3)
                        base_win_rate = calib_data[symbol].get("wfa_p", base_win_rate)
                        base_payout = calib_data[symbol].get("wfa_b", base_payout)
                        logger.info(f"[{symbol}] ✅ Calibração Ativa: Q={kalman_q}, R={kalman_r}, p={base_win_rate}, b={base_payout}")
                    else:
                        logger.warning(f"[{symbol}] ⚠️ Usando Defaults (Ativo não calibrado)")
            except Exception as e:
                logger.error(f"[{symbol}] Erro ao ler calibração: {e}")
        else:
            logger.warning(f"[{symbol}] ⚠️ Usando Defaults (Arquivo não encontrado)")


        self.kf = KalmanFilter1D(process_variance=kalman_q, measurement_variance=kalman_r)
        self.risk_manager = BayesianRiskManager(
            base_win_rate=base_win_rate,
            base_payout=base_payout,
            kelly_fraction=config['kelly_fraction'],
            tick_value=config['tick_value'],
            capital_allocation=self.allocation
        )
        
        self.n_states = config['n_states']
        self.hmm_model = None
        self.initialized = False

    def startup(self):
        """Treina o HMM para o ativo."""
        logger.info(f"[{self.symbol}] Inicializando...")
        if not self.engine.connect():
            return False
        
        # Coleta dados para treino (WFA janelas de 3000)
        df_init = self.engine.get_latest_m1_data(count=3000)
        if df_init is None or len(df_init) < 1000:
            logger.error(f"[{self.symbol}] Dados insuficientes para startup.")
            return False
        
        # Prep Kalman
        df_init['kalman'] = [self.kf.update(z) for z in df_init['close']]
        
        # Treino HMM
        try:
            # Adiciona jitter para evitar matrizes singulares em ativos de baixa vol (DI1)
            df_init['returns'] = df_init['kalman'].pct_change()
            df_init['vol'] = df_init['returns'].rolling(15).std()
            clean = df_init.dropna().copy()
            X = clean[['returns', 'vol']].values.copy()
            X += np.random.normal(0, 1e-9, X.shape)
            
            from hmmlearn.hmm import GaussianHMM
            self.hmm_model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100, random_state=42)
            self.hmm_model.fit(X)
            self.initialized = True
            logger.info(f"[OK] [{self.symbol}] HMM treinado.")
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] Falha no treino HMM: {e}")
            return False

    def process_tick(self):
        """Executa um ciclo M1 para o ativo."""
        if not self.initialized: return

        df = self.engine.get_latest_m1_data(count=50)
        if df is None: return

        current_price = df['close'].iloc[-1]
        kalman_val = self.kf.update(current_price)
        
        # Predict Regime and Confidence
        df['kalman'] = [self.kf.update(z) for z in df['close']]
        df['returns'] = df['kalman'].pct_change()
        df['vol'] = df['returns'].rolling(15).std()
        
        feat = df[['returns', 'vol']].dropna().iloc[-1:].values.copy()
        if len(feat) == 0: return
        feat_jitter = feat + np.random.normal(0, 1e-9, feat.shape)
        
        # Predict Regime and Posterior Probabilities (IA Confidence)
        regime = self.hmm_model.predict(feat_jitter)[0]
        probs = self.hmm_model.predict_proba(feat_jitter)[0]
        confidence = probs[regime]
        
        # ATR para Sizing
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                      abs(df['low'] - df['close'].shift(1))))
        atr = df['tr'].rolling(14).mean().iloc[-1]
        
        # Portfolio Kelly Sizing com Confiança Dinâmica
        contracts, risk_pct, debug = self.risk_manager.calculate_position_size(
            total_capital=self.capital_total,
            hmm_regime=regime,
            atr_points=atr,
            confidence=confidence
        )
        
        # Determinar Viés (Direção)
        bias = "COMPRA" if current_price > kalman_val else "VENDA"
        if contracts == 0:
            bias = "AGUARDAR"

        logger.info(f"| {self.symbol: <6} | {bias: <8} | Preço: {current_price: >8.2f} | Regime: {regime} | Lote: {contracts: >2} | {debug}")

        # Shadow Trading Execution & Management
        positions = mt5.positions_get(symbol=self.symbol, magic=self.engine.magic)
        
        if positions:
            # 1. SAÍDA POR MUDANÇA DE REGIME (HMM)
            if regime == 0:
                logger.info(f"[EXIT] {self.symbol} | Saída Antecipada: Fim do Regime de Volatilidade (Regime 0)")
                self.engine.close_all_positions()
                return

            # 2. TRAILING STOP VIA KALMAN
            pos = positions[0]
            if pos.type == mt5.ORDER_TYPE_BUY:
                if current_price < kalman_val:
                    logger.info(f"[EXIT] {self.symbol} | Saída Kalman: Crossover detectado (Price < Kalman)")
                    self.engine.close_all_positions()
            elif pos.type == mt5.ORDER_TYPE_SELL:
                if current_price > kalman_val:
                    logger.info(f"[EXIT] {self.symbol} | Saída Kalman: Crossover detectado (Price > Kalman)")
                    self.engine.close_all_positions()
        
        else:
            # 3. ENTRADA COM ESCUDO DE VOLATILIDADE (2x ATR)
            if contracts > 0 and regime != 0:
                side = mt5.ORDER_TYPE_BUY if bias == "COMPRA" else mt5.ORDER_TYPE_SELL
                sl_dist = atr * 2
                self.engine.execute_market_order(side, contracts, current_price, sl_points=sl_dist)

class PortfolioManager:
    """Gerenciador central do portfólio multi-ativo."""
    def __init__(self, capital=100000.0):
        self.capital = capital
        self.workers = {}
        self.running = False

    def startup(self):
        if not mt5.initialize():
            logger.error("[FAIL] Falha MT5")
            return False
            
        for symbol, cfg in PORTFOLIO_CONFIG.items():
            worker = AssetWorker(symbol, cfg, self.capital)
            if worker.startup():
                self.workers[symbol] = worker
            else:
                logger.warning(f"[WARN] Ignorando {symbol} devido a falha na inicialização.")
        
        return len(self.workers) > 0

    def run(self):
        self.running = True
        logger.info("[START] Portfolio Bot Iniciado (WDO, WSP, DI1)")
        last_minute = -1
        
        try:
            while self.running:
                now = datetime.now()
                if now.minute != last_minute:
                    # Rodamos o ciclo para todos os ativos
                    threads = []
                    for worker in self.workers.values():
                        t = threading.Thread(target=worker.process_tick)
                        threads.append(t)
                        t.start()
                    
                    for t in threads:
                        t.join()
                        
                    last_minute = now.minute
                
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        self.running = False
        for worker in self.workers.values():
            worker.engine.close_all_positions()
            worker.engine.shutdown()
        logger.info("[STOP] Portfolio Bot encerrado.")

if __name__ == "__main__":
    # Capital base de exemplo
    p_manager = PortfolioManager(capital=20000.0)
    if p_manager.startup():
        p_manager.run()
