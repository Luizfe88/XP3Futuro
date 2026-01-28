import functools
print = functools.partial(print, flush=True)
import json
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
import config
import utils
try:
    from fundamentals import fundamental_fetcher
except Exception:
    fundamental_fetcher = None

logger = logging.getLogger("bot")
print(f"[DEBUG] Verificando modelo em: {os.path.abspath('ml_trade_history.json')}", flush=True)
print(f"[DEBUG] Verificando modelo em: {os.path.abspath('qtable.npy')}", flush=True)

# Lazy ML dependencies
RF = GB = ET = RidgeCls = ScalerCls = KFoldCls = XGBRegressor = None
def ensure_ml_deps():
    global RF, GB, ET, RidgeCls, ScalerCls, KFoldCls, XGBRegressor
    if RF and RidgeCls and ScalerCls and KFoldCls:
        return True
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        print("[DEBUG] Importando sklearn.ensemble...", flush=True)
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
        print("[DEBUG] sklearn.ensemble importado.", flush=True)
        print("[DEBUG] Importando sklearn.linear_model...", flush=True)
        from sklearn.linear_model import Ridge
        print("[DEBUG] sklearn.linear_model importado.", flush=True)
        print("[DEBUG] Importando sklearn.preprocessing...", flush=True)
        from sklearn.preprocessing import StandardScaler
        print("[DEBUG] sklearn.preprocessing importado.", flush=True)
        print("[DEBUG] Importando sklearn.model_selection...", flush=True)
        from sklearn.model_selection import KFold
        print("[DEBUG] sklearn.model_selection importado.", flush=True)
        try:
            print("[DEBUG] Importando xgboost...", flush=True)
            import xgboost as xgb
            XGBRegressor = xgb.XGBRegressor
            print("[DEBUG] xgboost importado.", flush=True)
        except Exception:
            XGBRegressor = None
            print("[DEBUG] xgboost indisponível.", flush=True)
        RF, GB, ET = RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
        RidgeCls = Ridge
        ScalerCls = StandardScaler
        KFoldCls = KFold
        return True
    except Exception as e:
        print(f"[ERROR] Falha ao importar deps de ML: {e}", flush=True)
        return False

class EnsembleOptimizer:
    """
    Otimizador Ensemble avançado + Q-Learning adaptativo
    Aprende continuamente com trades reais para melhorar predições
    """
    def __init__(self, history_file="ml_trade_history.json", qtable_file="qtable.npy"):
        # === Ensemble ===
        self.models = {}
        if ensure_ml_deps():
            self.models = {
                'rf': RF(n_estimators=120, max_depth=8, random_state=42),
                'gb': GB(n_estimators=100, learning_rate=0.05, max_depth=6),
                'ridge': RidgeCls(alpha=1.5)
            }
            if XGBRegressor:
                self.models['xgb'] = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, subsample=0.8)
        else:
            self.models = {'ridge': None}
        self.ensemble_weights = {'rf': 0.35, 'gb': 0.25, 'xgb': 0.30, 'ridge': 0.10}
        self.scaler = ScalerCls() if ScalerCls else None
        self.ensemble_trained = False
        self.scaler_fitted = False
        self.models_stocks = None
        self.models_futures = None
        if 'rf' in self.models:
            logger.info("✅ RandomForest carregado")
        else:
            logger.warning("⚠️ RandomForest indisponível (deps ML ausentes)")
        
        self.history_file = history_file
        self.history = self.load_history()
        
        # === Q-Learning ===
        self.states = 10000  # ✅ AUMENTADO: 10000 estados (RSI 25 * ADX 20 * Vol 10 * Momentum 2)
        self.actions = 3
        self.q_table = np.zeros((self.states, self.actions))
        self.qtable_file = qtable_file
        self.load_qtable()
        
        self.alpha = 0.12  # Taxa de aprendizado levemente maior
        self.gamma = 0.95
        self.epsilon = 0.05
        self.epsilon_min = 0.008  # Mínimo menor
        self.epsilon_decay = 0.9995  # ✅ Decaimento mais lento
        
        self.last_state = None
        self.last_action = None
        
        # ✅ Treina ensemble no init se houver dados suficientes
        if len(self.history) >= 20:
            logger.info(f"🎯 Inicializando com {len(self.history)} trades históricos...")
            self.train_ensemble()
            
        # ✅ Garante arquivos iniciais
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        """Cria arquivos vazios se não existirem"""
        try:
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'w') as f:
                    json.dump([], f)
                logger.info(f"🆕 Arquivo de histórico criado: {self.history_file}")
            
            if not os.path.exists(self.qtable_file):
                np.save(self.qtable_file, self.q_table)
                logger.info(f"🆕 Arquivo Q-Table criado: {self.qtable_file}")
        except Exception as e:
            logger.error(f"Erro ao criar arquivos iniciais: {e}")

    def force_save(self):
        """Força salvamento de todos os dados"""
        try:
            logger.info("💾 Forçando salvamento de dados ML...")
            self.save_history()
            self.save_qtable()
            logger.info("✅ Dados ML salvos com sucesso.")
        except Exception as e:
            logger.error(f"❌ Erro no force_save: {e}")

    # ========================
    # ✅ NOVO: TREINAMENTO REAL DO ENSEMBLE
    # ========================
    def train_ensemble(self):
        """
        Treina todos os modelos do ensemble com histórico de trades reais
        Só executa se houver pelo menos 50 trades (reduzido de 100)
        """
        try:
            if len(self.history) < 20:
                logger.info(f"RF treino pulado: histórico insuficiente ({len(self.history)}/20 trades)")
                return
            
            df = pd.DataFrame(self.history)
            if df.empty:
                return
            df['asset_type'] = df.get('asset_type', '').fillna('')
            feats = pd.json_normalize(df['features'])
            feats = feats.fillna(0)
            feats['symbol'] = df['symbol']
            feats['asset_type'] = df['asset_type']
            base_cost = (getattr(config, 'B3_FEES_PCT', 0.0003) * 2) + getattr(config, 'AVG_SPREAD_PCT_DEFAULT', 0.001)
            slip_default = config.SLIPPAGE_MAP.get("DEFAULT", 0.0020)
            feats['costs_pct'] = feats['symbol'].apply(lambda s: config.SLIPPAGE_MAP.get(s, slip_default)) + base_cost
            y_all = df['pnl_pct'] - feats['costs_pct']
            cols_num = feats.select_dtypes(include=[np.number]).columns
            X_all = feats[cols_num].values
            if not KFoldCls or not self.scaler or not self.models:
                return
            X_scaled = self.scaler.fit_transform(X_all)
            self.scaler_fitted = True
            kf = KFoldCls(n_splits=5, shuffle=True, random_state=42)
            scores = {'rf': [], 'gb': [], 'xgb': [], 'ridge': []}
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
                for name, model in self.models.items():
                    model.fit(X_train, y_train)
                    scores[name].append(model.score(X_val, y_val))
            new_weights = {}
            total_score = 0
            for name, sc in scores.items():
                avg_score = np.mean(sc)
                weight = max(0.05, avg_score)
                new_weights[name] = weight
                total_score += weight
            for name in new_weights:
                new_weights[name] /= total_score
            self.ensemble_weights = new_weights
            self.ensemble_trained = True
            df_st = feats[feats['asset_type'] == 'STOCK']
            df_fu = feats[feats['asset_type'] == 'FUTURE']
            def _train_subset(df_src):
                if df_src.empty:
                    return None
                X = df_src[cols_num].values
                y = y_all.loc[df_src.index]
                if X.shape[0] < 20:
                    return None
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import Ridge
                sc = StandardScaler()
                Xs = sc.fit_transform(X)
                rf = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
                gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
                rg = Ridge(alpha=1.5)
                rf.fit(Xs, y)
                gb.fit(Xs, y)
                rg.fit(Xs, y)
                return {"scaler": sc, "rf": rf, "gb": gb, "ridge": rg, "cols": list(cols_num)}
            self.models_stocks = _train_subset(df_st)
            self.models_futures = _train_subset(df_fu)
            
            logger.info(f"✅ Ensemble Re-treinado! Novos pesos: {new_weights}")
            if 'rf' in self.models:
                logger.info("✅ RandomForest: treino concluído")
            
            # Treina modelo final com TODOS os dados
            for model in self.models.values():
                model.fit(X_scaled, y)
                
            self.save_ensemble_state()
            
            if getattr(config, 'ML_TRAIN_PER_SYMBOL', False) and 'symbol' in df.columns and 'rf' in self.models:
                counts = df['symbol'].value_counts()
                min_samples = getattr(config, 'ML_PER_SYMBOL_MIN_SAMPLES', 50)
                for sym, cnt in counts.items():
                    if cnt < min_samples:
                        continue
                    idx = df.index[df['symbol'] == sym].tolist()
                    X_sub = X_scaled[idx]
                    y_sub = y.iloc[idx]
                    if len(X_sub) < 10:
                        continue
                    logger.info(f"🏃 RandomForest: treino por ativo {sym} ({cnt} amostras)")
                    kf_sym = KFoldCls(n_splits=5, shuffle=True, random_state=42)
                    scores_sym = []
                    rf_params = self.models['rf'].get_params()
                    for train_i, val_i in kf_sym.split(X_sub):
                        rf_local = type(self.models['rf'])(**rf_params)
                        rf_local.fit(X_sub[train_i], y_sub.iloc[train_i])
                        scores_sym.append(rf_local.score(X_sub[val_i], y_sub.iloc[val_i]))
                    logger.info(f"✅ RandomForest: treino por ativo concluído {sym} (CV Score Médio: {np.mean(scores_sym):.4f})")

        except Exception as e:
            logger.error(f"Erro ao treinar ensemble ML: {e}")
            features_list = []
            targets = []
            
            for trade in self.history:
                if 'features' in trade and 'pnl_pct' in trade:
                    f = trade['features']
                    if isinstance(f, dict) and f.get('adx', 0) < 25:
                        continue
                    features_list.append(f)
                    slip_default = config.SLIPPAGE_MAP.get("DEFAULT", 0.0020)
                    base_cost = (getattr(config, 'B3_FEES_PCT', 0.0003) * 2) + getattr(config, 'AVG_SPREAD_PCT_DEFAULT', 0.001)
                    slip = config.SLIPPAGE_MAP.get(trade.get('symbol', 'DEFAULT'), slip_default)
                    targets.append(trade['pnl_pct'] - (base_cost + slip))
            
            if len(features_list) < 50:
                logger.warning(f"Features incompletas: {len(features_list)}/50")
                return
            
            # Converte para arrays
            df_features = pd.DataFrame(features_list).fillna(0)
            y = np.array(targets)

            # Remove colunas não numéricas se existirem
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            X = df_features[numeric_cols].values

            # ✅ Novo Check: Evita arrays vazios
            if X.shape[0] == 0 or X.shape[1] == 0:
                logger.warning("Dados inválidos para treinamento")
                return

            # ✅ Treina com Cross-Validation (5-fold)
            if not KFoldCls:
                logger.warning("KFold indisponível")
                return
            kf = KFoldCls(n_splits=5, shuffle=True, random_state=42)
            
            for name, model in self.models.items():
                try:
                    # Roda 5-fold CV e loga o score médio
                    scores = []
                    if name == 'rf':
                        logger.info("🏃 RandomForest: treino iniciado")
                    for train_index, val_index in kf.split(X_scaled):
                        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        model.fit(X_train, y_train)
                        scores.append(model.score(X_val, y_val))
                    
                    # Treina final no dado completo
                    model.fit(X_scaled, y)
                    logger.info(f"✅ Modelo {name} treinado (CV Score Médio: {np.mean(scores):.4f})")
                    if name == 'rf':
                        logger.info("✅ RandomForest: treino concluído")
                except Exception as e:
                    logger.error(f"Erro ao treinar modelo {name}: {e}")
            
            self.ensemble_trained = True
            logger.info(f"🧠 Ensemble RETREINADO com {len(self.history)} amostras (Cross-Validation OK)")
            
        except Exception as e:
            logger.error(f"Erro no treinamento do ensemble: {e}")

    # ========================
    # ✅ NOVO: PREDIÇÃO COM ENSEMBLE
    # ========================
    def predict_signal_score(self, features: dict) -> float:
        """
        Usa o ensemble treinado para prever score de um sinal
        Retorna score predito (pode ser usado como bônus no signal score)
        """
        try:
            if not self.ensemble_trained:
                return 0.0
            
            # Converte features para DataFrame
            df_feat = pd.DataFrame([features])
            numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
            X = df_feat[numeric_cols].fillna(0).values
            
            if not self.scaler_fitted:
                return 0.0
            
            X_scaled = self.scaler.transform(X)
            
            # Predição com cada modelo + pesos
            predictions = {}
            if 'rf' in self.models:
                logger.info("🤖 RandomForest: previsão executada")
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = pred
                except:
                    predictions[name] = 0.0
            
            # Combina predições com pesos
            score = sum(predictions[name] * self.ensemble_weights[name] 
                       for name in predictions.keys())
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Erro ao prever score: {e}")
            return 0.0

    # ========================
    # OTIMIZAÇÃO DE PARÂMETROS
    # ========================
    def optimize(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Otimiza parâmetros para um símbolo usando dados históricos
        
        Args:
            df: DataFrame com dados OHLCV
            symbol: Símbolo do ativo
        
        Returns:
            dict com parâmetros otimizados ou None se falhar
        """
        try:
            if df is None or len(df) < 100:
                logger.warning(f"Dados insuficientes para otimizar {symbol}")
                return None
            
            best_params = None
            best_score = -float('inf')
            
            # Grid search simplificado
            ema_short_range = [9, 12, 15, 18, 21]
            ema_long_range = [21, 34, 50, 89, 144]
            
            for ema_short in ema_short_range:
                for ema_long in ema_long_range:
                    if ema_short >= ema_long:
                        continue
                    
                    score = self._backtest_params(df, {
                        "ema_short": ema_short,
                        "ema_long": ema_long,
                        "rsi_low": 35,
                        "rsi_high": 65,
                        "adx_threshold": 20,
                        "mom_min": 0.001
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "ema_short": ema_short,
                            "ema_long": ema_long,
                            "rsi_low": 35,
                            "rsi_high": 65,
                            "adx_threshold": 20,
                            "mom_min": 0.001
                        }
            
            if best_params and best_score > 0:
                logger.info(f"✅ {symbol}: Parâmetros otimizados (score: {best_score:.2f})")
                return best_params
            else:
                logger.warning(f"⚠️ {symbol}: Otimização não melhorou parâmetros padrão")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao otimizar {symbol}: {e}")
            return None
    
    def _backtest_params(self, df: pd.DataFrame, params: dict) -> float:
        """
        Backtesta parâmetros e retorna score
        Score = Profit Factor * Win Rate
        """
        try:
            ema_short = df['close'].ewm(span=params['ema_short'], adjust=False).mean()
            ema_long = df['close'].ewm(span=params['ema_long'], adjust=False).mean()
            signals = (ema_short > ema_long).astype(int).diff()
            
            returns = []
            in_position = False
            entry_price = 0
            
            for i in range(1, len(df)):
                if signals.iloc[i] == 1 and not in_position:
                    entry_price = df['close'].iloc[i]
                    in_position = True
                elif signals.iloc[i] == -1 and in_position:
                    exit_price = df['close'].iloc[i]
                    ret = (exit_price - entry_price) / entry_price
                    returns.append(ret)
                    in_position = False
            
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            wins = returns_array[returns_array > 0]
            losses = returns_array[returns_array < 0]
            
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
            gross_profit = wins.sum() if len(wins) > 0 else 0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            score = profit_factor * win_rate * 100
            return score
            
        except Exception as e:
            logger.error(f"Erro no backtest: {e}")
            return 0.0

    # ========================
    # HISTÓRICO E PERSISTÊNCIA
    # ========================
    def load_history(self):
        """Carrega histórico de trades do disco"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📊 Histórico ML carregado: {len(data)} trades")
                    return data
            except Exception as e:
                logger.error(f"Erro ao carregar histórico ML: {e}")
        return []

    def save_history(self):
        """Salva histórico de trades no disco"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Erro ao salvar histórico ML: {e}")

    def load_qtable(self):
        """Carrega Q-Table do disco"""
        if os.path.exists(self.qtable_file):
            try:
                self.q_table = np.load(self.qtable_file)
                logger.info("🧠 Q-Table carregada")
            except Exception as e:
                logger.error(f"Erro ao carregar Q-Table: {e}")

    def save_qtable(self):
        """Salva Q-Table no disco"""
        try:
            np.save(self.qtable_file, self.q_table)
        except Exception as e:
            logger.error(f"Erro ao salvar Q-Table: {e}")

    # ========================
    # ✅ FEATURE ENGINEERING MELHORADO
    # ========================
    def extract_features(self, ind: dict, symbol: str, df: pd.DataFrame = None):
        """
        Extrai features avançadas dos indicadores para ML
        Agora com features adicionais para melhor aprendizado
        """
        try:
            if not isinstance(ind, dict):
                ind = {'close': ind if isinstance(ind, (int, float)) else 0}
            
            features = {}
            
            # Features básicas
            features['rsi'] = ind.get('rsi', 50)
            features['adx'] = ind.get('adx', 20)
            
            # ATR
            atr_val = ind.get('atr_real', 1.0)
            features['atr_pct'] = atr_val[-1] if isinstance(atr_val, (list, np.ndarray)) else atr_val
            
            # Volume
            features['volume_ratio'] = ind.get('volume_ratio', 1.0)
            
            # Trend
            features['ema_trend'] = 1 if ind.get('ema_fast', 0) > ind.get('ema_slow', 0) else -1
            
            # Condições de mercado
            features['macro_ok'] = 1 if ind.get('macro_trend_ok', False) else 0
            features['vol_breakout'] = 1 if ind.get('vol_breakout', False) else 0
            features['z_score_vol'] = ind.get('atr_zscore', 0)
            
            # ✅ NOVAS FEATURES
            features['rsi_distance_to_mid'] = abs(ind.get('rsi', 50) - 50)
            features['adx_strength'] = ind.get('adx', 20) / 50  # Normalizado
            features['momentum'] = ind.get('momentum', 0.0)
            
            # VWAP distance
            close_price = ind.get('close', 0)
            vwap = ind.get('vwap', close_price)
            if vwap and close_price:
                features['vwap_distance'] = abs(close_price - vwap) / close_price if close_price != 0 else 0
                features['dist_vwap'] = (close_price - vwap) / vwap if vwap != 0 else 0 # ✅ NOVO: dist_vwap signed
            else:
                features['vwap_distance'] = 0
                features['dist_vwap'] = 0
            
            # Score do sinal original
            features['time_score'] = ind.get('score', 0)
            
            # Temporal
            now = datetime.now()
            features['hour'] = now.hour
            features['day_of_week'] = now.weekday()
            
            # Market regime (se disponível)
            features['market_regime'] = ind.get('market_regime', 0)

            # ✅ NOVO: Performance histórica do símbolo
            symbol_trades = [t for t in self.history if t.get('symbol') == symbol]
            if symbol_trades:
                pnl_hist = [t['pnl_pct'] for t in symbol_trades[-20:]]
                features['pnl_hist_mean'] = np.mean(pnl_hist) if pnl_hist else 0
                features['pnl_hist_std'] = np.std(pnl_hist) if len(pnl_hist) > 1 else 0
                features['win_rate_hist'] = sum(1 for p in pnl_hist if p > 0) / len(pnl_hist) if pnl_hist else 0.5
            else:
                features['pnl_hist_mean'] = 0
                features['pnl_hist_std'] = 0
                features['win_rate_hist'] = 0.5

            is_fut = utils.is_future(symbol)
            if not is_fut:
                fund = fundamental_fetcher.get_fundamentals(symbol) if fundamental_fetcher else {}
                features['pe_ratio'] = fund.get('pe_ratio', 0.0)
                features['roe'] = fund.get('roe', 0.0)
                features['market_cap'] = fund.get('market_cap', 0.0) / 1e9
            else:
                try:
                    features['macro_selic'] = float(os.getenv("XP3_OVERRIDE_SELIC", "0.105") or 0.105)
                except Exception:
                    features['macro_selic'] = 0.105
                try:
                    vix_val = utils.get_vix_br()
                    features['vix'] = float(vix_val or 25.0)
                except Exception:
                    features['vix'] = 25.0

            # ✅ NOVO: SENTIMENT (Placeholder por enquanto, vindo do news_filter)
            from news_filter import get_news_sentiment
            features['sentiment_score'] = get_news_sentiment(symbol)
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair features: {e}")
            return {}

    # ========================
    # ✅ Q-LEARNING MELHORADO
    # ========================
    def discretize_state(self, ind: dict) -> int:
        """
        Discretiza estado com maior granularidade (10000 estados)
        Considera RSI, ADX, Volume Ratio e Momentum
        """
        try:
            # ✅ AUMENTADO: RSI (25) * ADX (20) * Vol (10) * Momentum (2) = 10000
            rsi_bucket = min(int(ind.get('rsi', 50) / 4), 24)  # 0-100 -> 25 buckets
            adx_bucket = min(int(ind.get('adx', 20) / 5), 19)  # 0-100 -> 20 buckets
            vol_bucket = min(int(ind.get('volume_ratio', 1.0) * 5), 9)  # 0-2.0 -> 10 buckets
            momentum_bucket = 1 if ind.get('momentum', 0) > 0 else 0  # 2 buckets
            
            state = rsi_bucket * 400 + adx_bucket * 20 + vol_bucket * 2 + momentum_bucket
            state = min(state, self.states - 1)
            
            return state
            
        except Exception as e:
            logger.error(f"Erro ao discretizar estado: {e}")
            return 0

    def choose_action(self, state: int) -> int:
        """Escolhe ação usando epsilon-greedy"""
        try:
            if np.random.random() < self.epsilon:
                return np.random.randint(self.actions)
            return int(np.argmax(self.q_table[state]))
        except Exception as e:
            logger.error(f"Erro ao escolher ação: {e}")
            return 0  # HOLD em caso de erro

    def get_ml_signal(self, ind: dict) -> str:
        """Obtém sinal do Q-Learning"""
        try:
            state = self.discretize_state(ind)
            action = self.choose_action(state)
            
            self.last_state = state
            self.last_action = action
            
            actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            return actions_map[action]
            
        except Exception as e:
            logger.error(f"Erro ao obter sinal ML: {e}")
            return "HOLD"

    def update_qlearning(self, reward: float, next_ind: dict):
        """Atualiza Q-Table com novo reward"""
        try:
            if self.last_state is None or self.last_action is None:
                return
            
            next_state = self.discretize_state(next_ind)
            best_next = np.max(self.q_table[next_state])
            
            old_val = self.q_table[self.last_state, self.last_action]
            self.q_table[self.last_state, self.last_action] += self.alpha * (
                reward + self.gamma * best_next - old_val
            )
            
            self.save_qtable()
            
        except Exception as e:
            logger.error(f"Erro ao atualizar Q-Learning: {e}")

    def decay_epsilon(self):
        """Decai epsilon para reduzir exploração ao longo do tempo"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ========================
    # ✅ REGISTRO E APRENDIZADO
    # ========================
    def record_trade(self, symbol: str, pnl_pct: float, indicators: dict):
        """
        Registra resultado de trade e atualiza ML
        ✅ Agora com reward contínuo e treinamento automático
        """
        try:
            # Extrai features
            features = self.extract_features(indicators, symbol)
            
            # Registra trade
            trade_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'symbol': symbol,
                'pnl_pct': pnl_pct,
                'features': features,
                'asset_type': 'FUTURE' if utils.is_future(symbol) else 'STOCK'
            }
            
            self.history.append(trade_data)
            
            # Limita histórico
            if len(self.history) > 5000:
                self.history.pop(0)
            
            self.save_history()
            
            # ✅ REWARD PARA CONSISTÊNCIA (não apenas profit)
            # Combina: PnL + bônus por consistência + penalidade por variância
            base_reward = np.tanh(pnl_pct * 10)  # +3% → ~0.99
            
            # Bônus por consistência (trades pequenos positivos são bons)
            if 0 < pnl_pct <= 1.5:
                consistency_bonus = 0.2  # Recompensa trades pequenos mas positivos
            elif pnl_pct > 1.5:
                consistency_bonus = 0.1  # Trades grandes são bons mas não tão consistentes
            elif pnl_pct < -2.0:
                consistency_bonus = -0.3  # Penaliza perdas grandes
            else:
                consistency_bonus = 0.0
            
            reward = base_reward + consistency_bonus
            reward = max(-1.0, min(1.0, reward))  # Clamp
            
            # Atualiza Q-Learning
            self.update_qlearning(reward, indicators)
            self.decay_epsilon()
            
            # ✅ LOG MAIS INFORMATIVO
            logger.info(
                f"💾 ML Atualizado | {symbol} | PnL: {pnl_pct:+.2f}% | "
                f"Reward: {reward:+.2f} (base:{base_reward:+.2f} +cons:{consistency_bonus:+.2f}) | "
                f"Epsilon: {self.epsilon:.4f}"
            )
            
            # ✅ TREINA A CADA N TRADES (Conforme configurado)
            self.trade_counter = getattr(self, 'trade_counter', 0) + 1
            
            if self.trade_counter % config.ML_RETRAIN_THRESHOLD == 0:
                self.train_ensemble()
                logger.info(f"🔄 Ensemble retreinado após {self.trade_counter} trades")
        
        except Exception as e:
            logger.error(f"Erro ao registrar trade no ML: {e}")

# Instância global
ml_optimizer = EnsembleOptimizer()
