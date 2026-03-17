import time
import logging
import logging.handlers
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import threading
from datetime import datetime
from execution_engine import ExecutionEngine
from risk_validation import BayesianRiskManager
from hmm_validation import KalmanFilter1D, train_and_plot_hmm
from utils import resolve_symbol
import json
import os
import sys
import io
from sklearn.preprocessing import StandardScaler

# Force UTF-8 for standard output/error to handle emojis on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# 1. CONFIGURAÇÃO DE LOGGING E CORES
# ==========================================
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.handlers.TimedRotatingFileHandler(
            "logs/portfolio_bot.log", 
            when="H", 
            interval=4, 
            backupCount=18, 
            encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PortfolioBot")

# ==========================================
# 2. DEFINIÇÕES DO PORTFÓLIO
# ==========================================
PORTFOLIO_CONFIG = {
    "WDO$N": {
        "allocation": 0.50, 
        "tick_value": 5.0, 
        "base_win_rate": 0.55, 
        "base_payout": 1.5,
        "n_states": 2,
        "kelly_fraction": 0.10
    },
    "WIN$N": {
        "allocation": 0.50, 
        "tick_value": 0.20, 
        "base_win_rate": 0.54, 
        "base_payout": 1.5,
        "n_states": 2,
        "kelly_fraction": 0.08
    }
}
class AssetWorker:
    """Instância individual para cada ativo com Troca de Regimes Soberana."""
    def __init__(self, symbol, config, capital_total):
        self.symbol = symbol
        self.resolved_symbol = resolve_symbol(symbol)
        self.allocation = config['allocation']
        self.capital_total = capital_total
        self.engine = ExecutionEngine(symbol=self.resolved_symbol, magic_number=999000 + list(PORTFOLIO_CONFIG.keys()).index(symbol))
        
        self.timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30
        }
        
        # Estrutura de Perfis Triplos
        self.profiles = {}
        self.current_q = {tf: 1e-4 for tf in self.timeframes}
        self.current_r = {tf: 1e-3 for tf in self.timeframes}
        
        base_win_rate = config['base_win_rate']
        base_payout = config['base_payout']
        
        if os.path.exists("calibrated_assets.json"):
            try:
                with open("calibrated_assets.json", "r") as f:
                    calib_data = json.load(f)
                    if symbol in calib_data:
                        symbol_data = calib_data[symbol]
                        self.profiles = symbol_data.get("profiles", {})
                        
                        # Set initial defaults from TREND profile
                        for tf in self.timeframes:
                            if tf in self.profiles:
                                trend = self.profiles[tf].get("TREND", {})
                                self.current_q[tf] = trend.get("kalman_q", 1e-4)
                                self.current_r[tf] = trend.get("kalman_r", 1e-3)
                                if tf == "M5":
                                    base_win_rate = trend.get("wr", base_win_rate)
                                    base_payout = trend.get("payout", base_payout)
                        
                        logger.info(f"[{symbol}] ✅ Perfis Triplos (TREND, SIDEWAYS, PROTECTION) Carregados.")
                    else:
                        logger.warning(f"[{symbol}] ⚠️ Usando Defaults (Ativo não calibrado)")
            except Exception as e:
                logger.error(f"[{symbol}] Erro ao ler calibração: {e}")

        self.kfs = {tf: KalmanFilter1D(process_variance=self.current_q[tf], measurement_variance=self.current_r[tf]) for tf in self.timeframes}
        self.risk_manager = BayesianRiskManager(
            base_win_rate=base_win_rate,
            base_payout=base_payout,
            kelly_fraction=config['kelly_fraction'],
            tick_value=config['tick_value'],
            capital_allocation=self.allocation
        )
        
        self.n_states = 3 # Evolução para 3 estados
        self.hmm_models = {tf: None for tf in self.timeframes}
        self.scalers = {tf: StandardScaler() for tf in self.timeframes}
        self.regime_maps = {tf: {} for tf in self.timeframes}
        self.initialized = {tf: False for tf in self.timeframes}

    def startup(self):
        """Treina os HMMs e identifica os regimes soberanos."""
        logger.info(f"[{self.symbol}] Inicializando Regimes Soberanos...")
        if not self.engine.connect():
            return False
            
        all_initialized = True
        for tf_name, tf_val in self.timeframes.items():
            df_init = self.engine.get_latest_data(tf_val, count=3000)
            if df_init is None or len(df_init) < 1000:
                all_initialized = False
                continue
            
            df_init['kalman'] = [self.kfs[tf_name].update(z) for z in df_init['close']]
            
            try:
                df_init['returns'] = df_init['kalman'].pct_change()
                df_init['vol'] = df_init['returns'].rolling(15).std()
                clean = df_init.dropna().copy()
                X = clean[['returns', 'vol']].values.copy()
                X += np.random.normal(0, 1e-9, X.shape)
                
                X_scaled = self.scalers[tf_name].fit_transform(X)
                
                from hmmlearn.hmm import GaussianHMM
                self.hmm_models[tf_name] = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100, random_state=42, min_covar=1e-2)
                self.hmm_models[tf_name].fit(X_scaled)
                
                # Mapeamento do Regime (Soberania por Volatilidade)
                vols = self.hmm_models[tf_name].means_[:, 1]
                sorted_vols = np.argsort(vols)
                self.regime_maps[tf_name] = {
                    sorted_vols[0]: 0, # SIDEWAYS
                    sorted_vols[1]: 1, # TREND
                    sorted_vols[2]: 2  # PROTECTION
                }
                
                self.initialized[tf_name] = True
                logger.info(f"[OK] [{self.symbol}] HMM 3-States treinado no {tf_name}.")
            except Exception as e:
                logger.error(f"[{self.symbol}] Falha no startup {tf_name}: {e}")
                all_initialized = False

        return all_initialized

    def process_tick(self):
        """Loop de SCAN com Troca de Parâmetros e Kelly Dinâmico."""
        if not all(self.initialized.values()): return

        regimes = {}
        confidences = {}
        current_price = None
        atr_m5 = None

        # 1. Identificar Regimes e Trocar Parâmetros em Tempo Real
        for tf_name, tf_val in self.timeframes.items():
            df = self.engine.get_latest_data(tf_val, count=50)
            if df is None: return

            close_price = df['close'].iloc[-1]
            if tf_name == "M5":
                current_price = close_price
                df['tr'] = np.maximum(df['high'] - df['low'], 
                                     np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                               abs(df['low'] - df['close'].shift(1))))
                atr_m5 = df['tr'].rolling(14).mean().iloc[-1]
            
            # Predição do Regime
            df['kalman_tmp'] = [self.kfs[tf_name].update(z) for z in df['close']]
            df['returns'] = df['kalman_tmp'].pct_change()
            df['vol'] = df['returns'].rolling(15).std()
            
            feat = df[['returns', 'vol']].dropna().iloc[-1:].values.copy()
            if len(feat) == 0: return
            
            try:
                feat_scaled = self.scalers[tf_name].transform(feat + np.random.normal(0, 1e-9, feat.shape))
                raw_regime = self.hmm_models[tf_name].predict(feat_scaled)[0]
                regime = self.regime_maps[tf_name][raw_regime]
                regimes[tf_name] = regime
                confidences[tf_name] = self.hmm_models[tf_name].predict_proba(feat_scaled)[0][raw_regime]
                
                # --- TROCA DE PARÂMETROS KALMAN (Soberania) ---
                profile_name = "TREND" if regime == 1 else ("SIDEWAYS" if regime == 0 else "PROTECTION")
                if tf_name in self.profiles and profile_name in self.profiles[tf_name]:
                    p = self.profiles[tf_name][profile_name]
                    self.kfs[tf_name].Q = p.get("kalman_q", self.kfs[tf_name].Q)
                    self.kfs[tf_name].R = p.get("kalman_r", self.kfs[tf_name].R)
            except Exception as e:
                regimes[tf_name] = 0
                confidences[tf_name] = 0.5

        if current_price is None or atr_m5 is None: return

        # 2. Ajuste Dinâmico de Kelly
        m5_regime = regimes["M5"]
        # Perfil Moderado (KELLY=0.7)
        kelly_mult = 0.7 if m5_regime == 1 else (0.3 if m5_regime == 2 else 0.0)
        
        contracts, _, debug = self.risk_manager.calculate_position_size(
            total_capital=self.capital_total,
            hmm_regime=1, # Forçamos calculo base como trend
            atr_points=atr_m5,
            confidence=confidences["M5"]
        )
        
        if m5_regime == 0:
            contracts = 0
            debug += " | [BLOCK] Sideways Regime"
        elif m5_regime == 2:
            debug += " | [REDUCED] Protection Regime"

        # 3. Refinamento do Consenso Ensemble (Perfil Moderado)
        # Se M5 e M15 confirmam Regime 1 com WR > 60%, pode boletar mesmo que M30 esteja em Regime 0
        wr_m5 = confidences.get("M5", 0)
        wr_m15 = confidences.get("M15", 0)
        
        is_trend_consensus = (regimes["M5"] == 1) and (regimes["M15"] == 1)
        
        # Flexibilização: M30 Sideways (0) é permitido se M5/M15 TREND (1) e WR > 60%
        m30_ok = (regimes.get("M30", 0) == 1) or (regimes.get("M30", 0) == 0 and wr_m5 > 0.60 and wr_m15 > 0.60)
        
        # 🆕 Kalman Tolerance (Pullback Check)
        # Tolerância de 0.6 ATR (0.5 base + 20%)
        kf_val = self.kfs["M5"].x_hat
        dist_kf = abs(current_price - kf_val)
        max_dist = atr_m5 * 0.6
        kalman_ready = dist_kf <= max_dist
        
        can_trade = False
        if m5_regime == 1 and is_trend_consensus and m30_ok and kalman_ready:
            can_trade = True
        elif m5_regime == 2 and contracts > 0:
            can_trade = True # Permite entradas pequenas em exaustão
        
        if not can_trade and contracts > 0:
            contracts = 0
            reason = "No Trend Consensus" if not is_trend_consensus else ("M30 Block" if not m30_ok else "Kalman Dist")
            debug += f" | [BLOCKED] {reason}"

        # 4. Saída Soberana (M15 e M30 PROTECTION)
        positions = mt5.positions_get(symbol=self.symbol, magic=self.engine.magic)
        if positions:
            # Se o M15 ou M30 detectarem Regime 2 (Exaustão), fecha tudo
            if regimes["M15"] == 2 or regimes.get("M30") == 2:
                origin = "M30" if regimes.get("M30") == 2 else "M15"
                logger.info(f"🛡️ {LogColors.RED}[SOVEREIGN EXIT]{LogColors.RESET} {self.symbol} | {origin} detectou Exaustão (Regime 2)")
                self.engine.close_all_positions()
                return

            if regimes["M5"] == 0:
                logger.info(f"🚪 {LogColors.YELLOW}[EXIT]{LogColors.RESET} {self.symbol} | M5 em Consolidação (Regime 0)")
                self.engine.close_all_positions()
                return

        # Execução
        bias = "COMPRA" # Exemplo fixo, deve vir de lógica de direção
        bias_label = f"{LogColors.GREEN}🟢 {bias}{LogColors.RESET}" if contracts > 0 else f"{LogColors.GRAY}⏳ AGUARDAR{LogColors.RESET}"
        
        def format_regime(r):
            colors = {0: LogColors.YELLOW, 1: LogColors.GREEN, 2: LogColors.RED}
            return f"{colors.get(r, '')}{r}{LogColors.RESET}"

        logger.info(
            f"🔍 {LogColors.BOLD}[SCAN]{LogColors.RESET} "
            f"| {LogColors.CYAN}{self.symbol:<6}{LogColors.RESET} "
            f"| {bias_label:<18} "
            f"| 📊 Regimes: M5({format_regime(regimes['M5'])}), M15({format_regime(regimes['M15'])}) "
            f"| 📦 Lote: {LogColors.BOLD}{contracts:>2}{LogColors.RESET} "
            f"| {LogColors.GRAY}{debug}{LogColors.RESET}"
        )

        if contracts > 0 and not positions:
            side = mt5.ORDER_TYPE_BUY if bias == "COMPRA" else mt5.ORDER_TYPE_SELL
            self.engine.execute_market_order(side, contracts, current_price, sl_points=atr_m5*2)

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
        logger.info(f"🚀 {LogColors.BOLD}[START]{LogColors.RESET} Portfolio Bot Iniciado (10 Símbolos Ativos)")
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
        logger.info(f"🛑 {LogColors.RED}[STOP]{LogColors.RESET} Portfolio Bot encerrado.")

if __name__ == "__main__":
    # Capital base de exemplo
    p_manager = PortfolioManager(capital=20000.0)
    if p_manager.startup():
        p_manager.run()
