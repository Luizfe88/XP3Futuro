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
import utils  # Para acessar indicadores e funções auxiliares
import config
import logging

logger = logging.getLogger("ml")

class EnsembleOptimizer:
    """
    Otimizador Ensemble avançado + Q-Learning simples
    """
    def __init__(self, history_file="ml_trade_history.json", qtable_file="qtable.npy"):
        # === Ensemble ===
        self.models = {
            'rf': RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6),
            'xgb': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8),
            'ridge': Ridge(alpha=1.5)
        }
        self.ensemble_weights = {'rf': 0.35, 'gb': 0.25, 'xgb': 0.30, 'ridge': 0.10}
        self.scaler = StandardScaler()
        self.history_file = history_file
        self.history = self.load_history()
        
        # === Q-Learning ===
        self.states = 100  # 10 RSI buckets x 10 ADX buckets
        self.actions = 3   # 0=HOLD, 1=BUY, 2=SELL
        self.q_table = np.zeros((self.states, self.actions))
        self.qtable_file = qtable_file
        self.load_qtable()
        
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.15  # Exploração inicial
        self.last_state = None
        self.last_action = None
        
        # Decaimento de epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # ========================
    # HISTÓRICO E TREINAMENTO
    # ========================
    def load_history(self):
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
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Erro ao salvar histórico ML: {e}")

    def load_qtable(self):
        if os.path.exists(self.qtable_file):
            try:
                self.q_table = np.load(self.qtable_file)
                logger.info("🧠 Q-Table carregada")
            except:
                pass

    def save_qtable(self):
        try:
            np.save(self.qtable_file, self.q_table)
        except Exception as e:
            logger.error(f"Erro ao salvar Q-Table: {e}")

    # ========================
    # FEATURE ENGINEERING (AVANÇADO MAS REALISTA)
    # ========================
    def extract_features(self, ind: dict, symbol: str, df: pd.DataFrame = None):
        features = {}
        
        # Básicos do seu indicador atual
        features['rsi'] = ind.get('rsi', 50)
        features['adx'] = ind.get('adx', 20)
        features['atr_pct'] = ind.get('atr_real', 1.0)
        features['volume_ratio'] = ind.get('volume_ratio', 1.0)
        features['ema_trend'] = 1 if ind.get('ema_fast', 0) > ind.get('ema_slow', 0) else -1
        features['macro_ok'] = 1 if ind.get('macro_trend_ok', False) else 0
        features['vol_breakout'] = 1 if ind.get('vol_breakout', False) else 0
        features['z_score_vol'] = ind.get('atr_zscore', 0)
        
        # Time-based
        now = datetime.now()
        features['hour'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_power_hour'] = 1 if utils.is_power_hour() else 0
        
        # Regime de mercado
        features['risk_off'] = 1 if utils.detect_market_regime() == "RISK_OFF" else 0
        
        # Candlestick patterns simples
        if df is not None and len(df) >= 2:
            prev = df.iloc[-2]
            curr = df.iloc[-1]
            body = abs(curr['close'] - curr['open'])
            range_ = curr['high'] - curr['low']
            features['is_doji'] = 1 if body <= range_ * 0.1 else 0
            # Engulfing simplificado
            bullish_engulf = (curr['close'] > curr['open']) and (prev['close'] < prev['open']) and (curr['open'] < prev['close']) and (curr['close'] > prev['open'])
            bearish_engulf = (curr['close'] < curr['open']) and (prev['close'] > prev['open']) and (curr['open'] > prev['close']) and (curr['close'] < prev['open'])
            features['engulfing'] = 1 if bullish_engulf else -1 if bearish_engulf else 0
        
        # Volatilidade recente vs média
        if 'atr_real' in ind and len(ind.get('atr_real', [])) > 20:
            recent_vol = ind['atr_real']
            features['vol_expansion'] = recent_vol / np.mean(ind['atr_real'][-20:]) if np.mean(ind['atr_real'][-20:]) > 0 else 1
        
        return features

    # ========================
    # RECORD TRADE (para ensemble)
    # ========================
    def record_trade(self, symbol: str, ind: dict, profit_pct: float, df=None):
        features = self.extract_features(ind, symbol, df)
        features['symbol'] = symbol
        features['profit_pct'] = profit_pct
        self.history.append(features)
        self.save_history()
        
        if len(self.history) >= 20 and len(self.history) % 15 == 0:
            self.retrain_ensemble()
            self.decay_epsilon()

    def retrain_ensemble(self):
        if len(self.history) < 30:
            return
        
        df_hist = pd.DataFrame([h for h in self.history if 'profit_pct' in h])
        feature_cols = [c for c in df_hist.columns if c not in ['profit_pct', 'symbol']]
        X = df_hist[feature_cols].values
        y = df_hist['profit_pct'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        predictions = []
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Atualiza pesos com base na performance recente (simples)
        logger.info("🧠 Ensemble re-treinado")
        
    # ========================
    # Q-LEARNING
    # ========================
    def discretize_state(self, ind: dict) -> int:
        rsi_bucket = min(int(ind.get('rsi', 50) / 10), 9)
        adx_bucket = min(int(ind.get('adx', 20) / 10), 9)
        state = rsi_bucket * 10 + adx_bucket
        return state

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        return int(np.argmax(self.q_table[state]))

    def get_ml_signal(self, ind: dict) -> str:
        """Retorna 'BUY', 'SELL' ou 'HOLD' baseado no Q-Learning"""
        state = self.discretize_state(ind)
        action = self.choose_action(state)
        
        self.last_state = state
        self.last_action = action
        
        actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return actions_map[action]

    def update_qlearning(self, reward: float, next_ind: dict):
        if self.last_state is None or self.last_action is None:
            return
        
        next_state = self.discretize_state(next_ind)
        best_next = np.max(self.q_table[next_state])
        
        old_val = self.q_table[self.last_state, self.last_action]
        self.q_table[self.last_state, self.last_action] += self.alpha * (
            reward + self.gamma * best_next - old_val
        )
        
        self.save_qtable()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Instância global
ml_optimizer = EnsembleOptimizer()