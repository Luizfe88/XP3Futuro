import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from collections import deque
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger("ml")

class EnsembleOptimizer:
    """
    Otimizador Ensemble avançado + Q-Learning adaptativo
    Aprende continuamente com trades reais para melhorar predições
    """
    def __init__(self, history_file="ml_trade_history.json", qtable_file="qtable.npy"):
        # === Ensemble ===
        self.models = {
            'rf': RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6),
            'xgb': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, subsample=0.8),
            'ridge': Ridge(alpha=1.5)
        }
        self.ensemble_weights = {'rf': 0.35, 'gb': 0.25, 'xgb': 0.30, 'ridge': 0.10}
        self.scaler = StandardScaler()
        self.ensemble_trained = False
        self.scaler_fitted = False
        
        self.history_file = history_file
        self.history = self.load_history()
        
        # === Q-Learning ===
        self.states = 1000  # ✅ Aumentado de 100 para 1000
        self.actions = 3
        self.q_table = np.zeros((self.states, self.actions))
        self.qtable_file = qtable_file
        self.load_qtable()
        
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # ✅ Mais lento (era 0.995)
        
        self.last_state = None
        self.last_action = None
        
        # ✅ Treina ensemble no init se houver dados suficientes
        if len(self.history) >= 50:
            logger.info(f"🎯 Inicializando com {len(self.history)} trades históricos...")
            self.train_ensemble()

    # ========================
    # ✅ NOVO: TREINAMENTO REAL DO ENSEMBLE
    # ========================
    def train_ensemble(self):
        """
        Treina todos os modelos do ensemble com histórico de trades reais
        Só executa se houver pelo menos 50 trades
        """
        try:
            if len(self.history) < 50:
                logger.debug(f"Histórico insuficiente para treinar: {len(self.history)}/50 trades")
                return
            
            # Cria DataFrame com features de cada trade
            features_list = []
            targets = []
            
            for trade in self.history:
                if 'features' in trade and 'pnl_pct' in trade:
                    features_list.append(trade['features'])
                    targets.append(trade['pnl_pct'])
            
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

            # ✅ Aplica StandardScaler (apenas uma vez)
            if not self.scaler_fitted:
                X_scaled = self.scaler.fit_transform(X)
                self.scaler_fitted = True
            else:
                X_scaled = self.scaler.transform(X)
            
            # ✅ Treina todos os modelos do ensemble
            for name, model in self.models.items():
                try:
                    model.fit(X_scaled, y)
                    logger.debug(f"✅ Modelo {name} treinado")
                except Exception as e:
                    logger.error(f"Erro ao treinar modelo {name}: {e}")
            
            self.ensemble_trained = True
            logger.info(f"🧠 Ensemble RETREINADO com {len(self.history)} amostras")
            
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
            else:
                features['vwap_distance'] = 0
            
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
                pnl_hist = [t['pnl_pct'] for t in symbol_trades[-20:]] if symbol_trades else []  # ✅ Proteção: if symbol_trades else [] (evita errors se vazio)
                features['pnl_hist_mean'] = np.mean(pnl_hist) if pnl_hist else 0
                features['pnl_hist_std'] = np.std(pnl_hist) if len(pnl_hist) > 1 else 0
                features['win_rate_hist'] = sum(1 for p in pnl_hist if p > 0) / len(pnl_hist) if pnl_hist else 0.5
            else:
                features['pnl_hist_mean'] = 0
                features['pnl_hist_std'] = 0
                features['win_rate_hist'] = 0.5                 
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair features: {e}")
            return {}

    # ========================
    # ✅ Q-LEARNING MELHORADO
    # ========================
    def discretize_state(self, ind: dict) -> int:
        """
        Discretiza estado com maior granularidade (1000 estados)
        Considera RSI, ADX e Volume Ratio
        """
        try:
            # ✅ Buckets mais finos (5 em 5)
            rsi_bucket = min(int(ind.get('rsi', 50) / 5), 19)  # 0-95 → 19 buckets
            adx_bucket = min(int(ind.get('adx', 20) / 5), 19)  # 0-95 → 19 buckets
            vol_bucket = min(int(ind.get('volume_ratio', 1.0) * 10), 9)  # 0-2.0 → 9 buckets
            
            # Estado combinado: 20*20*10 = 4000 possíveis (limitamos a 1000)
            state = rsi_bucket * 100 + adx_bucket * 10 + vol_bucket
            state = min(state, self.states - 1)  # Garante que não exceda limite
            
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
                'features': features
            }
            
            self.history.append(trade_data)
            
            # Limita histórico
            if len(self.history) > 5000:
                self.history.pop(0)
            
            self.save_history()
            
            # ✅ REWARD CONTÍNUO (não mais binário)
            reward = np.tanh(pnl_pct * 10)  # +3% → ~0.99, -3% → ~-0.99
            
            # Atualiza Q-Learning
            self.update_qlearning(reward, indicators)
            self.decay_epsilon()
            
            # ✅ LOG MAIS INFORMATIVO
            logger.info(
                f"💾 ML Atualizado | {symbol} | PnL: {pnl_pct:+.2f}% | "
                f"Reward: {reward:+.2f} | Epsilon: {self.epsilon:.3f}"
            )
            
            # ✅ TREINA A CADA 10 TRADES (não todo trade)
            self.trade_counter = getattr(self, 'trade_counter', 0) + 1
            
            if self.trade_counter % 10 == 0:
                self.train_ensemble()
                logger.info(f"🔄 Ensemble retreinado após {self.trade_counter} trades")
        
        except Exception as e:
            logger.error(f"Erro ao registrar trade no ML: {e}")

# Instância global
ml_optimizer = EnsembleOptimizer()