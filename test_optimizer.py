import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from optimizer_optuna import backtest_params_on_df

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_cache(symbol):
    try:
        # Tenta carregar do parquet cached pelo otimizador_semanal
        # Assumindo estrutura padrão de diretórios
        path = f"data_cache/{symbol}_M15.parquet"
        try:
            df = pd.read_parquet(path)
            print(f"✅ Dados carregados do cache: {len(df)} linhas")
            return df
        except FileNotFoundError:
            print(f"⚠️ Cache não encontrado em {path}. Tentando conectar MT5...")
            
        if not mt5.initialize():
            print("❌ Falha inicialização MT5")
            return None
            
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 3000)
        if rates is None:
            print(f"❌ Falha ao baixar dados {symbol}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df.set_index('time', inplace=True)
        return df
        
    except Exception as e:
        print(f"❌ Erro carga dados: {e}")
        return None

def run_debug():
    symbol = "PETR4"
    print(f"🚀 Iniciando debug isolado para {symbol}...")
    
    df = load_data_from_cache(symbol)
    if df is None:
        return

    # Parâmetros "Boms" (Relaxados)
    params = {
        "ema_short": 9, 
        "ema_long": 21,
        "rsi_low": 40,      # Relaxado
        "rsi_high": 60,     # Relaxado
        "adx_threshold": 15,# Relaxado
        "sl_atr_multiplier": 2.5,
        "tp_mult": 2.5, # tp_mult será calculado se tp_ratio for passado, mas aqui passamos direto
        "base_slippage": 0.001,
        "enable_shorts": 1
    }
    
    # Simular ML model None (vai usar fallback ou 0.85 fixo se forcei no codigo)
    print("\n🔬 Executando backtest_params_on_df...")
    metrics = backtest_params_on_df(symbol, params, df, ml_model=None)
    
    print("\n📊 RESULTADOS:")
    for k, v in metrics.items():
        if k != "equity_curve":
            print(f"  {k}: {v}")
            
    # Validar se trades > 0
    if metrics.get("total_trades", 0) == 0:
        print("\n❌ SEM TRADES! Verifique:")
        print("   - Lógica de entrada (Momentum/RSI)")
        print("   - Filtro de ML (se hardcoded)")
        print("   - Verificação de Equity/Margin")
    else:
        print(f"\n✅ TRADES GERADOS: {metrics['total_trades']}")

if __name__ == "__main__":
    run_debug()
