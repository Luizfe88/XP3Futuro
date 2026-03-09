import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import os
import time

# ==========================================
# 1. CONFIGURAÇÕES DOS ATIVOS (B3)
# ==========================================
ASSET_METADATA = {
    "WIN$N": {"tick_size": 5.0, "tick_value": 1.0, "point_value": 0.20},
    "IND$N": {"tick_size": 5.0, "tick_value": 5.0, "point_value": 1.0},
    "WDO$N": {"tick_size": 0.5, "tick_value": 5.0, "point_value": 10.0},
    "DOL$N": {"tick_size": 0.5, "tick_value": 25.0, "point_value": 50.0},
    "CCM$N": {"tick_size": 0.01, "tick_value": 4.50, "point_value": 450.0},
    "BGI$N": {"tick_size": 0.05, "tick_value": 16.5, "point_value": 330.0},
    "ICF$N": {"tick_size": 0.05, "tick_value": 5.0, "point_value": 100.0},
    "BIT$N": {"tick_size": 10.0, "tick_value": 1.0, "point_value": 0.10},
    "DI1$N": {"tick_size": 0.001, "tick_value": 1.0, "point_value": 1000.0},
    "WSP$N": {"tick_size": 0.5, "tick_value": 25.0, "point_value": 50.0}
}

def tf_name(tf):
    if tf == mt5.TIMEFRAME_M1: return "M1"
    if tf == mt5.TIMEFRAME_M5: return "M5"
    return str(tf)

# ==========================================
# 2. COMPONENTES MATEMÁTICOS
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
# 3. MOTOR DE VALIDAÇÃO BATCH
# ==========================================

def validate_asset(symbol, df, train_size=None, test_size=None):
    """Executa o pipeline completo para um ativo específico com janelas adaptativas."""
    print(f"\n--- VALIDANDO: {symbol} ---")
    
    kf = KalmanFilter1D()
    df['kalman'] = [kf.update(z) for z in df['close']]
    
    total_bars = len(df)
    current_idx = 0
    oos_results = []
    tick_val = ASSET_METADATA.get(symbol, {}).get('tick_value', 1.0)
    
    # Ajuste adaptativo de janelas
    if train_size is None or test_size is None:
        if total_bars >= 5000:
            train_size, test_size = 3000, 500
        elif total_bars >= 2000:
            train_size, test_size = 1200, 300
        else:
            train_size, test_size = 500, 100
            
    print(f"   📊 Janelas: Train={train_size}, Test={test_size} (Total={total_bars} bars)")
    
    while (current_idx + train_size + test_size) <= total_bars:
        is_end = current_idx + train_size
        df_is = df.iloc[current_idx:is_end].copy()
        
        oos_end = is_end + test_size
        df_oos = df.iloc[is_end:oos_end].copy()
        
        # Treino IS
        df_is['returns'] = df_is['kalman'].pct_change()
        df_is['vol'] = df_is['returns'].rolling(15).std()
        is_clean = df_is.dropna().copy()
        
        if len(is_clean) < 100: 
            print(f"      ⚠️ Bloco {current_idx} ignorado: is_clean muito pequeno ({len(is_clean)})")
            current_idx += test_size
            continue

        try:
            # Adiciona jitter (ruído infinitesimal) para evitar matrizes singulares
            X = is_clean[['returns', 'vol']].values.copy()
            X += np.random.normal(0, 1e-9, X.shape)
            
            model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X)
            trend_regime = np.argmin(model.means_[:, 1])
            
            # Execução OOS
            df_oos['returns'] = df_oos['kalman'].pct_change()
            df_oos['vol'] = df_oos['returns'].rolling(15).std()
            oos_clean = df_oos.dropna().copy()
            
            if not oos_clean.empty:
                X_oos = oos_clean[['returns', 'vol']].values
                oos_regimes = model.predict(X_oos)
                period_pnl = 0.0
                for i in range(len(oos_clean)):
                    if oos_regimes[i] == trend_regime:
                        points_change = (oos_clean.iloc[i]['close'] - oos_clean.iloc[i]['open'])
                        tick_size = ASSET_METADATA.get(symbol, {}).get('tick_size', 1.0)
                        ticks = points_change / tick_size
                        period_pnl += ticks * tick_val
                
                oos_results.append(period_pnl)
        except Exception as e:
            print(f"      ❌ Erro no bloco IS {current_idx}: {e}")
            pass
            
        current_idx += test_size
        
    print(f"   📉 Fim da validação: {len(oos_results)} resultados OOS.")
    return oos_results

def main():
    terminal_path = r"C:\MetaTrader 5 Terminal\terminal64.exe"
    if not mt5.initialize(path=terminal_path):
        print("❌ Falha MT5")
        return

    symbols = ['IND$N','WSP$N','WDO$N','DOL$N','CCM$N','BGI$N','ICF$N','BIT$N','DI1$N']
    report = []

    print(f"📡 Iniciando extração e validação para {len(symbols)} ativos...")
    
    for symbol in symbols:
        # Garante que o símbolo está no Market Watch
        if not mt5.symbol_select(symbol, True):
            # Tenta fallback sem o 'N' se necessário (Ex: IND$ instead of IND$N)
            fallback = symbol.replace('$N', '$')
            if not mt5.symbol_select(fallback, True):
                print(f"⚠️ {symbol}: Não encontrado ou indisponível no terminal.")
                continue
            symbol = fallback

        # Tenta baixar os dados (M1 primeiro, M5 como fallback)
        rates = None
        timeframes = [(mt5.TIMEFRAME_M1, 5000), (mt5.TIMEFRAME_M5, 3000)]
        
        for tf, count in timeframes:
            for attempt in range(2):
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
                if rates is not None and len(rates) >= 800: # Mínimo absoluto para algum WFA
                    break
                time.sleep(2)
            if rates is not None and len(rates) >= 800:
                print(f"   ✅ {symbol}: Dados obtidos ({tf_name(tf)} - {len(rates)} bars)")
                break
            
        if rates is None or len(rates) < 800:
            actual_len = len(rates) if rates is not None else 0
            print(f"   ⚠️ {symbol}: Falha na extração. ({actual_len} bars)")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        results = validate_asset(symbol, df)
        
        if results:
            total_pnl = sum(results)
            win_rate = len([r for r in results if r > 0]) / len(results)
            report.append({
                'Symbol': symbol,
                'Total_PnL': total_pnl,
                'WFA_Success_Rate': f"{win_rate:.2%}",
                'Status': "✅ OK" if total_pnl > 0 else "⚠️ Fraco"
            })
            print(f"   PnL Total OOS: R$ {total_pnl:.2f} | Taxa: {win_rate:.2%}")
        else:
            print(f"   ⚠️ {symbol}: Dados insuficientes para WFA.")

    mt5.shutdown()
    
    # Exibe Relatório Final
    df_report = pd.DataFrame(report)
    print("\n" + "="*50)
    print("🏆 RELATÓRIO FINAL DE PORTFÓLIO QUANT 🏆")
    print("="*50)
    print(df_report.to_string(index=False))
    print("="*50)
    
    # Save Report
    df_report.to_csv("logs/multi_asset_report.csv", index=False)
    print("\nRelatório salvo em 'logs/multi_asset_report.csv'")

if __name__ == "__main__":
    main()
