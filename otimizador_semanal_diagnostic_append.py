
# =========================================================
# 6. DIAGNOSTIC FORCE SIGNALS
# =========================================================
def diagnostic_force_signals(df: pd.DataFrame):
    """
    Cria um DF manipulado onde 10% dos dados têm preço caindo e RSI baixo
    para FORÇAR sinais de compra e testar a mecânica de execução.
    """
    df_forced = df.copy()
    n = len(df_forced)
    
    # Criar 5 zonas de manipulação
    for i in range(5):
        start = int(n * (0.1 + i*0.2))
        end = start + 50
        if end >= n: break
        
        # Forçar queda bruta no preço (Drop 5%)
        df_forced.iloc[start:end, df_forced.columns.get_loc('close')] *= 0.95
        df_forced.iloc[start:end, df_forced.columns.get_loc('low')] *= 0.95
        df_forced.iloc[start:end, df_forced.columns.get_loc('high')] *= 0.95
        
        # Forçar RSI baixo indiretamente (não adianta só mudar o indicador calculado se for recalculado lá dentro)
        # O backtest recalcula RSI. Entao precisamos mudar o preço mesmo.
        # Ao dropar o preço em sequencia, o RSI vai cair.
        
    return df_forced

if __name__ == "__main__":
    print("🚀 MODO DIAGNÓSTICO INICIADO")
    # Teste rápido com PETR4
    if not ensure_mt5_connection():
        print("❌ Sem conexão MT5")
        exit()
        
    test_symbol = "PETR4"
    print(f"📥 Carregando dados de {test_symbol}...")
    df = load_data_with_retry(test_symbol, 2000, mt5.TIMEFRAME_M15)
    
    if df is not None:
        print(f"📊 Dados Carregados: {len(df)} candles")
        
        # 1. Teste Normal
        print("\n--- TESTE 1: DADOS REAIS ---")
        res_real = helper_optimize_symbol(test_symbol)
        
        # 2. Teste Forçado
        print("\n--- TESTE 2: DADOS FORÇADOS (FAIL-SAFE CHECK) ---")
        df_forced = diagnostic_force_signals(df)
        
        # Precisamos chamar o backtest direto pois helper_optimize baixa dados de novo
        from optimizer_optuna import backtest_params_on_df
        
        dummy_params = {
            "ema_short": 9, "ema_long": 21,
            "rsi_low": 40, "rsi_high": 70,
            "adx_threshold": 10,
            "sl_atr_multiplier": 2.5,
            "tp_mult": 5.0,
            "base_slippage": 0.0,
            "enable_shorts": 1
        }
        
        metrics = backtest_params_on_df(test_symbol, dummy_params, df_forced)
        print(f"🏁 Resultado Forçado: Trades={metrics['total_trades']} | WR={metrics['win_rate']:.2f}")
    
    else:
        print("❌ Falha ao carregar dados de teste.")
