import logging
import json
import os
import argparse
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from hmmlearn.hmm import GaussianHMM

from config_futures import FUTURES_CONFIGS
from hmm_validation import KalmanFilter1D

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Calibration")

CALIBRATION_FILE = "calibrated_assets.json"

class WalkForwardValidator:
    """Minimal WFA Motor just to extract p and b."""
    def __init__(self, data, q, r, tick_value, slippage_cost, train_window_size=3000, test_window_size=500):
        self.data = data
        self.q = q
        self.r = r
        self.tick_value = tick_value
        self.slippage_cost = slippage_cost
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.oos_results = []
    
    def run(self):
        total_bars = len(self.data)
        current_idx = 0
        winning_trades = 0
        losing_trades = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        kf = KalmanFilter1D(process_variance=self.q, measurement_variance=self.r)
        # Pre-calculate Kalmans to save time
        self.data['kalman'] = [kf.update(z) for z in self.data['close']]

        while (current_idx + self.train_window_size + self.test_window_size) <= total_bars:
            is_end = current_idx + self.train_window_size
            df_is = self.data.iloc[current_idx:is_end].copy()
            oos_end = is_end + self.test_window_size
            df_oos = self.data.iloc[is_end:oos_end].copy()
            
            # --- In-Sample HMM ---
            df_is['returns'] = df_is['kalman'].pct_change()
            df_is['vol'] = df_is['returns'].rolling(15).std()
            is_clean = df_is.dropna(subset=['returns', 'vol'])
            
            if is_clean.empty:
               current_idx += self.test_window_size
               continue

            X_is = is_clean[['returns', 'vol']].values
            X_is += np.random.normal(0, 1e-9, X_is.shape)

            try:
                model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_is)
                trend_regime = np.argmin(model.means_[:, 1])
            except Exception as e:
                current_idx += self.test_window_size
                continue

            # --- Out-Of-Sample Execution ---
            df_oos['returns'] = df_oos['kalman'].pct_change()
            df_oos['vol'] = df_oos['returns'].rolling(15).std()
            oos_clean = df_oos.dropna(subset=['returns', 'vol'])
            
            if not oos_clean.empty:
                X_oos = oos_clean[['returns', 'vol']].values
                X_oos += np.random.normal(0, 1e-9, X_oos.shape)
                try:
                    oos_regimes = model.predict(X_oos)
                    
                    for i in range(len(oos_clean)):
                        regime = oos_regimes[i]
                        if regime == trend_regime:
                            # Simplistic trade assumption: If close > open, we make points equivalent to the bar body
                            points_diff = oos_clean.iloc[i]['close'] - oos_clean.iloc[i]['open']
                            financial_diff = (points_diff * self.tick_value)
                            
                            trade_pnl = financial_diff - self.slippage_cost
                            
                            if trade_pnl > 0:
                                winning_trades += 1
                                gross_profit += trade_pnl
                            elif trade_pnl < 0:
                                losing_trades += 1
                                gross_loss += abs(trade_pnl)
                except Exception as e:
                    pass

            current_idx += self.test_window_size
            
        total_trades = winning_trades + losing_trades
        if total_trades == 0:
            return 0.5, 1.0 # fallback

        win_rate = winning_trades / total_trades
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        payout_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Cap limits to prevent extreme Kelly
        win_rate = min(max(win_rate, 0.2), 0.8)
        payout_ratio = min(max(payout_ratio, 0.5), 5.0)

        return win_rate, payout_ratio


class AssetCalibrator:
    def __init__(self, symbol):
        self.symbol = symbol
        self_config = FUTURES_CONFIGS.get(symbol, {})
        if not self_config:
            logger.warning(f"No config found in config_futures for {symbol}. Using defaults.")
            self.tick_value = 0.20
            self.slippage_base = 2.0
        else:
            specs = self_config.get('specs', {})
            self.tick_value = specs.get('value_per_tick', 1.0) / specs.get('tick_size', 1.0) # Normalizing point value roughly
            
            # Extract slippage points/ticks and convert to financial cost
            slip_map = specs.get('slippage_base', {})
            base_slip_points = slip_map.get('avg', 1.0)
            self.slippage_base = base_slip_points * specs.get('value_per_tick', 1.0) / specs.get('tick_size', 1.0)

    def calibrate(self):
        logger.info(f"🚀 Iniciando Calibração Completa para {self.symbol}...")
        
        terminal_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
        if not mt5.initialize(path=terminal_path):
            logger.error("❌ Falha ao inicializar MT5")
            return

        # Busca dados H1 e M1 (para WFA longo)
        logger.info(f"📡 Extraindo dados de {self.symbol}...")
        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 15000)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            logger.error(f"❌ Falha ao extrair dados para {self.symbol}")
            return
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # 1. Kalman Tuning
        # Procuramos o par Q, R que minimize a divergência mantendo smoothness.
        best_q, best_r = self.tune_kalman(df)
        logger.info(f"✅ Kalman Tuning completado: Q={best_q}, R={best_r}")
        
        # 2. HMM & WFA para Expectativa
        logger.info(f"⏳ Executando Walk-Forward Analysis (WFA) para descobrir p e b...")
        wfa = WalkForwardValidator(df.copy(), q=best_q, r=best_r, 
                                   tick_value=self.tick_value, 
                                   slippage_cost=self.slippage_base)
        
        win_rate, payout_ratio = wfa.run()
        
        logger.info(f"✅ WFA completado: p={win_rate:.2f}, b={payout_ratio:.2f}")
        
        self.save_calibration(best_q, best_r, win_rate, payout_ratio)


    def tune_kalman(self, df):
        """Simplistic heuristic grid search for Kalman params based on variance tests."""
        # For simplicity, returning stable hard-coded values logic depending on asset type, 
        # or conducting a small evaluation of var/lag.
        
        prices = df['close'].values[-2000:]
        configs = [
            (1e-5, 1e-3),
            (1e-4, 1e-3),
            (1e-3, 1e-3),
            (1e-4, 1e-2)
        ]
        
        best_score = float('inf')
        best_cfg = (1e-4, 1e-3)
        
        for q, r in configs:
            kf = KalmanFilter1D(process_variance=q, measurement_variance=r)
            purified = np.array([kf.update(z) for z in prices])
            
            # Penalty for lag (diff to price) + Penalty for noise (variance of diff)
            diffs = prices - purified
            lag_penalty = np.mean(np.abs(diffs))
            noise_penalty = np.std(purified)
            
            score = (lag_penalty * 0.5) + (noise_penalty * 0.5)
            if score < best_score:
                best_score = score
                best_cfg = (q, r)
                
        return best_cfg[0], best_cfg[1]

    def save_calibration(self, q, r, p, b):
        filepath = CALIBRATION_FILE
        data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except:
                pass
                
        data[self.symbol] = {
            "kalman_q": q,
            "kalman_r": r,
            "wfa_p": round(p, 4),
            "wfa_b": round(b, 4),
            "tick_value_base": self.tick_value,
            "slippage_cost_base": self.slippage_base,
            "calibrated_at": pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
        logger.info(f"💾 Calibração do ativo {self.symbol} salva com sucesso em {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Ativo a calibrar, ex: 'WIN$N' ou múltiplos separados por vírgula 'WIN$N,WDO$N,CCM$N'")
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbol.split(',')]
    
    for symbol in symbols:
        logger.info(f"--- Iniciando calibração em lote para: {symbol} ---")
        calibrator = AssetCalibrator(symbol)
        calibrator.calibrate()
        logger.info(f"--- Fim da calibração para: {symbol} ---\n")
