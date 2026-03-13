import logging
import json
import os
import argparse
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config_futures import FUTURES_CONFIGS
from hmm_validation import KalmanFilter1D

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Calibration")

CALIBRATION_FILE = "calibrated_assets.json"

class WalkForwardValidator:
    """WFA Motor evoluído para 3 regimes (0, 1, 2)."""
    def __init__(self, data, q, r, tick_value, slippage_cost, train_window_size=3000, test_window_size=500):
        self.data = data
        self.q = q
        self.r = r
        self.tick_value = tick_value
        self.slippage_cost = slippage_cost
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        
    def run(self):
        n_windows = max(1, (len(self.data) - self.train_window_size) // self.test_window_size)
        total_bars = len(self.data)
        current_idx = 0
        
        # Metrics per regime
        # 0: Sideways, 1: Trend, 2: Protection
        regime_metrics = {
            0: {"total": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0},
            1: {"total": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0},
            2: {"total": 0, "wins": 0, "pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
        }
        
        kf = KalmanFilter1D(process_variance=self.q, measurement_variance=self.r)
        self.data['kalman'] = [kf.update(z) for z in self.data['close']]

        while (current_idx + self.train_window_size + self.test_window_size) <= total_bars:
            is_end = current_idx + self.train_window_size
            df_is = self.data.iloc[current_idx:is_end].copy()
            oos_end = is_end + self.test_window_size
            df_oos = self.data.iloc[is_end:oos_end].copy()
            
            # --- In-Sample HMM (3 States) ---
            df_is['returns'] = df_is['kalman'].pct_change()
            df_is['vol'] = df_is['returns'].rolling(15).std()
            is_clean = df_is.dropna(subset=['returns', 'vol'])
            
            if is_clean.empty or len(is_clean) < 100:
               current_idx += self.test_window_size
               continue

            X_is = is_clean[['returns', 'vol']].values
            X_is += np.random.normal(0, 1e-9, X_is.shape)
            scaler = StandardScaler()
            X_is_scaled = scaler.fit_transform(X_is)

            try:
                # Evolução: 3 Estados
                model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42, min_covar=1e-3)
                model.fit(X_is_scaled)
                
                # Identificação Soberana por Volatilidade
                vols = model.means_[:, 1]
                sorted_vols = np.argsort(vols)
                regime_map = {
                    sorted_vols[0]: 0, # SIDEWAYS (Low Vol)
                    sorted_vols[1]: 1, # TREND (Mid Vol)
                    sorted_vols[2]: 2  # PROTECTION (High Vol)
                }
            except Exception:
                current_idx += self.test_window_size
                continue

            # --- Out-Of-Sample Execution ---
            df_oos['returns'] = df_oos['kalman'].pct_change()
            df_oos['vol'] = df_oos['returns'].rolling(15).std()
            oos_clean = df_oos.dropna(subset=['returns', 'vol'])
            
            if not oos_clean.empty:
                X_oos = oos_clean[['returns', 'vol']].values
                X_oos += np.random.normal(0, 1e-9, X_oos.shape)
                X_oos_scaled = scaler.transform(X_oos)
                try:
                    raw_regimes = model.predict(X_oos_scaled)
                    oos_regimes = [regime_map[r] for r in raw_regimes]

                    in_trade = False
                    entry_price = None
                    last_regime = None
                    
                    for i in range(len(oos_regimes)):
                        regime = oos_regimes[i]
                        price = oos_clean.iloc[i]['close']
                        
                        if not in_trade:
                            # Inicia trade em qualquer regime para medir performance
                            in_trade = True
                            entry_price = price
                            last_regime = regime
                        elif regime != last_regime:
                            # Fecha quando muda o regime (Troca Dinâmica)
                            trade_pnl = (price - entry_price) * self.tick_value - self.slippage_cost
                            met = regime_metrics[last_regime]
                            met["total"] += 1
                            met["pnl"] += trade_pnl
                            if trade_pnl > 0:
                                met["wins"] += 1
                                met["gross_profit"] += trade_pnl
                            else:
                                met["gross_loss"] += abs(trade_pnl)
                            
                            # Re-abre no novo regime
                            entry_price = price
                            last_regime = regime
                            
                    # Fecha trade final
                    if in_trade:
                        price = oos_clean.iloc[-1]['close']
                        trade_pnl = (price - entry_price) * self.tick_value - self.slippage_cost
                        met = regime_metrics[last_regime]
                        met["total"] += 1
                        met["pnl"] += trade_pnl
                        if trade_pnl > 0:
                            met["wins"] += 1
                            met["gross_profit"] += trade_pnl
                        else:
                            met["gross_loss"] += abs(trade_pnl)
                except Exception:
                    pass

            current_idx += self.test_window_size
            
        final_results = {}
        for r, m in regime_metrics.items():
            if m["total"] < 5:
                final_results[r] = {"wr": 0.0, "payout": 1.0, "total": m["total"]}
                continue
                
            wr = m["wins"] / m["total"]
            avg_win = m["gross_profit"] / m["wins"] if m["wins"] > 0 else 0
            loss_count = m["total"] - m["wins"]
            avg_loss = m["gross_loss"] / loss_count if loss_count > 0 else 1.0
            payout = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            final_results[r] = {
                "wr": round(wr, 4),
                "payout": round(payout, 4),
                "total": m["total"],
                "pnl": round(m["pnl"], 2)
            }
            
        return final_results


class AssetCalibrator:
    def __init__(self, symbol):
        self.symbol = symbol
        self_config = FUTURES_CONFIGS.get(symbol, {})
        if not self_config:
            logger.warning(f"No config found in config_futures for {self.symbol}. Using defaults.")
            self.tick_value = 0.20
            self.slippage_base = 2.0
            self.margin = 1000.0
        else:
            specs = self_config.get('specs', {}) # Wait, config_futures might have different structure
            # Re-checking config_futures structure from previous view_file (if I had it)
            # Based on futures_optimizer view: tick_size, point_value, fees, margin
            self.tick_size = self_config.get('tick_size', 1.0)
            self.point_value = self_config.get('point_value', 1.0)
            self.tick_value = self.point_value # points to financial
            self.slippage_base = self_config.get('slippage_base', {}).get('avg', 10) * self.point_value
            self.margin = self_config.get('margin', 1000.0)

    def calibrate(self):
        logger.info(f"🚀 Iniciando Triple Calibração (TREND, SIDEWAYS, PROTECTION) para {self.symbol}...")
        
        terminal_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
        if not mt5.initialize(path=terminal_path):
            logger.error("❌ Falha ao inicializar MT5")
            return

        timeframes = ["M5", "M15"] # Foco nos principais para velocidade
        all_calibrations = {}

        for tf_name in timeframes:
            tf_val = mt5.TIMEFRAME_M5 if tf_name == "M5" else mt5.TIMEFRAME_M15
            logger.info(f"📡 Extraindo dados de {tf_name} para {self.symbol}...")
            rates = mt5.copy_rates_from_pos(self.symbol, tf_val, 0, 15000)
            
            if rates is None or len(rates) < 2000:
                logger.error(f"❌ Dados insuficientes para {self.symbol} no {tf_name}.")
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Triple Grid Search: Um set de Q/R para cada objetivo de regime
            regimes_params = {}
            
            # 1. TREND: Maximizar WR e Payout
            logger.info(f"🔍 Otimizando TREND (Regime 1) para {tf_name}...")
            regimes_params["TREND"] = self.tune_for_regime(df, target_regime=1)
            
            # 2. SIDEWAYS: Maximizar estabilidade (Maior R para ignorar ruído)
            logger.info(f"🔍 Otimizando SIDEWAYS (Regime 0) para {tf_name}...")
            regimes_params["SIDEWAYS"] = self.tune_for_regime(df, target_regime=0)
            
            # 3. PROTECTION: Reatividade máxima (Menor R)
            logger.info(f"🔍 Otimizando PROTECTION (Regime 2) para {tf_name}...")
            regimes_params["PROTECTION"] = self.tune_for_regime(df, target_regime=2)
            
            all_calibrations[tf_name] = regimes_params
            
        mt5.shutdown()
        
        if all_calibrations:
            self.save_calibration(all_calibrations)

    def tune_for_regime(self, df, target_regime):
        """Busca Q/R que melhor performam ou se comportam no regime alvo."""
        # Grid reduzido para não demorar tando (Triple Search = 3x tempo)
        q_options = [1e-5, 1e-4, 5e-4, 1e-3]
        r_options = [1e-4, 1e-3, 1e-2, 5e-2]
        
        best_score = -999999.0
        best_params = {"kalman_q": 1e-4, "kalman_r": 1e-3, "wr": 0.0, "payout": 0.0}
        
        for q in q_options:
            for r in r_options:
                wfa = WalkForwardValidator(df.copy(), q, r, self.tick_value, self.slippage_base)
                results = wfa.run()
                
                res = results.get(target_regime)
                if not res: continue
                
                # Fitness Function baseada no regime
                if target_regime == 1: # TREND: WR + Payout
                    score = res["wr"] * 100 + res["payout"] * 20
                elif target_regime == 0: # SIDEWAYS: Queremos detectar MUITO (total alto) com PnL estável
                    score = res["total"] * 0.1 - abs(res["pnl"]) * 0.001
                else: # PROTECTION: Queremos Payout alto (saída rápida de perdedores)
                    score = res["payout"] * 50 + res["wr"] * 10
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "kalman_q": q,
                        "kalman_r": r,
                        "wr": res["wr"],
                        "payout": res["payout"],
                        "total_trades": res["total"]
                    }
        
        return best_params

    def save_calibration(self, all_calibrations):
        filepath = CALIBRATION_FILE
        data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except: pass
                
        data[self.symbol] = {
            "calibrated_at": pd.Timestamp.now().isoformat(),
            "profiles": all_calibrations
        }
        
        # Log de Desempenho Histórico por Regime
        logger.info(f"\n--- RELATÓRIO DE CALIBRAÇÃO: {self.symbol} ---")
        for tf, profiles in all_calibrations.items():
            t = profiles["TREND"]
            s = profiles["SIDEWAYS"]
            p = profiles["PROTECTION"]
            logger.info(f"[{tf}] TREND WR: {t['wr']:.1%} | SIDEWAYS Block Rate (Total): {s['total_trades']} | PROTECTION Payout: {p['payout']:.2f}")

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"💾 Calibração Tripla salva em {filepath}")


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
