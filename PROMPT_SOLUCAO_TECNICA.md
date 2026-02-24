# 🔧 PROMPT DE SOLUÇÃO TÉCNICA: Bot Trading B3 Multi-Asset

## 📋 CONTEXTO

Você é um arquiteto de software sênior especializado em sistemas de trading algorítmico. Sua missão é refatorar um bot de trading da B3 que atualmente **NÃO consegue operar simultaneamente em índice futuro (WIN/WDO) e mercado de ações (PETR4, VALE3, etc)** devido a 12 problemas arquiteturais críticos identificados.

**Objetivo:** Criar uma arquitetura que permita operação **profissional, escalável e simultânea** em múltiplas classes de ativos com diferentes características (margem, volatilidade, horários, lotes).

---

## 🎯 REQUISITOS FUNCIONAIS

### RF1: Asset Class Manager (Gerenciador de Classes de Ativos)

**Implementar um sistema que trate cada classe de ativo como entidade independente:**

```python
class AssetClassManager:
    """
    Gerencia configurações específicas por classe de ativo.
    Elimina tratamento genérico que causa conflitos.
    """
    
    def __init__(self):
        self.asset_classes = {
            'FUTURES_INDEX': FuturesIndexConfig(),
            'FUTURES_CURRENCY': FuturesCurrencyConfig(),
            'STOCKS_BLUE_CHIP': StocksBlueChipConfig(),
            'STOCKS_SMALL_CAP': StocksSmallCapConfig(),
        }
    
    def get_config(self, symbol: str) -> AssetConfig:
        """
        Retorna configuração específica baseada no símbolo.
        
        Parâmetros que DEVEM ser específicos:
        - risk_per_trade (% diferente para cada classe)
        - position_sizing_method (margem vs linear)
        - timeframe_primary (M5 para futuros, M15 para ações)
        - timeframe_confirmation (M15 para futuros, H1 para ações)
        - trading_hours (09:00-17:50 futuros, 10:00-17:55 ações)
        - max_positions (3 contratos WIN vs 8 ações)
        - capital_allocation_pct (dinâmico baseado em volatilidade)
        - sl_calculation_method (pontos vs percentual vs ATR)
        - min_tick_profit (50 pontos WIN vs 0.5% ações)
        - slippage_model (ticks para futuros vs % para ações)
        """
        pass
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                                sl_price: float, account_equity: float) -> float:
        """
        Calcula tamanho da posição considerando:
        
        FUTUROS:
        - Margem requerida (não exposição notional)
        - Valor do ponto (WIN R$1, WDO R$10)
        - Margem disponível real
        - Alavancagem implícita
        
        AÇÕES:
        - Exposição linear (R$ comprado = R$ exposto)
        - Lote mínimo (100 ações)
        - Capital disponível direto
        """
        pass
```

**Pontos críticos:**

1. **Capital Allocation dinâmico:**
   - Base: Futuros 35%, Ações 65%
   - Ajustar em real-time baseado em:
     - Volatilidade atual (VIX Brasil)
     - Margem livre real (não teórica)
     - Número de posições abertas
     - Horário (aumentar ações na power hour)

2. **Risk Calculation separado:**
   ```python
   # ERRADO (atual):
   risk_money = volume * entry_price * 0.025  # genérico
   
   # CERTO (proposto):
   if is_future:
       risk_money = (entry_price - sl_price) / point_value * point_size * volume
   else:
       risk_money = volume * entry_price * (1 - sl_price/entry_price)
   ```

---

### RF2: Multi-Timeframe Engine (Motor de Múltiplos Timeframes)

**Implementar confirmação MTF que realmente funcione:**

```python
class MultiTimeframeEngine:
    """
    Sincroniza análise de múltiplos timeframes por classe de ativo.
    Garante que confirmações sejam checadas ANTES de entrar.
    """
    
    def validate_entry(self, symbol: str, side: str, 
                      base_timeframe: str) -> Tuple[bool, str]:
        """
        Validação hierárquica obrigatória:
        
        FUTUROS (WIN/WDO):
        Primário: M5 (scalping)
        Confirmação: M15 (tendência)
        Filtro: H1 (regime)
        
        AÇÕES (PETR4/VALE3):
        Primário: M15 (swing)
        Confirmação: H1 (tendência)
        Filtro: D1 (macro)
        
        Lógica:
        1. Calcula indicadores em TODOS os timeframes
        2. Verifica alinhamento progressivo (de cima para baixo)
        3. Se H1 diz COMPRA e M15 diz VENDA -> REJEITA
        4. Se H1 diz COMPRA e M15 diz COMPRA -> APROVA
        5. Se H1 neutro e M15 forte -> APROVA com risco reduzido
        """
        
        # Hierarquia de confirmação
        hierarchy = self._get_hierarchy(symbol)
        
        signals = {}
        for tf in hierarchy:
            signals[tf] = self._calculate_signal(symbol, tf)
        
        # Validação cascata
        conflicts = self._check_conflicts(signals, side)
        if conflicts:
            return False, f"Conflito MTF: {conflicts}"
        
        # Score de alinhamento
        alignment_score = self._calculate_alignment(signals, side)
        if alignment_score < 0.70:  # 70% alinhamento mínimo
            return False, f"Alinhamento insuficiente: {alignment_score:.0%}"
        
        return True, f"MTF OK (score: {alignment_score:.0%})"
    
    def _check_conflicts(self, signals: dict, intended_side: str) -> list:
        """
        Detecta contradições entre timeframes.
        
        Exemplos de conflito:
        - H1 em baixa forte (ADX>30, EMA bearish) mas M15 quer comprar
        - D1 em range (ADX<20) mas M15 quer operar tendência
        - M5 sobre-esticado (RSI>80) mas quer comprar
        """
        pass
```

**Integração no fluxo principal:**

```python
# bot.py - fast_loop
def try_enter_position(symbol: str, side: str):
    # 1. PRIMEIRA VALIDAÇÃO: MTF
    mtf_ok, mtf_reason = multi_tf_engine.validate_entry(symbol, side, base_tf)
    
    if not mtf_ok:
        reject_and_log("MTF_Conflict", mtf_reason)
        return False
    
    # 2. Depois continua com outros filtros
    # ... resto do código
```

---

### RF3: Intelligent Filter Chain (Cadeia de Filtros Inteligente)

**Problema atual:** 12 filtros sequenciais com taxa de aprovação de 2.9%

**Solução:** Sistema de filtros **adaptativos e priorizados**

```python
class FilterChain:
    """
    Cadeia de filtros com priorização e bypass inteligente.
    """
    
    def __init__(self):
        self.filters = {
            # OBRIGATÓRIOS (nunca fazem bypass)
            'capital_available': {
                'priority': 1,
                'mandatory': True,
                'bypass_conditions': None
            },
            'mtf_alignment': {
                'priority': 2,
                'mandatory': True,
                'bypass_conditions': None
            },
            'risk_limits': {
                'priority': 3,
                'mandatory': True,
                'bypass_conditions': None
            },
            
            # ADAPTATIVOS (podem fazer bypass)
            'ml_confidence': {
                'priority': 4,
                'mandatory': False,
                'bypass_conditions': {
                    'high_adx': lambda ind: ind['adx'] > 40,
                    'strong_trend': lambda ind: ind['ema_diff'] > 0.02,
                    'high_volume': lambda ind: ind['volume_ratio'] > 2.0
                },
                'bypass_threshold': 2  # Precisa de 2 condições
            },
            'anti_chop': {
                'priority': 5,
                'mandatory': False,
                'bypass_conditions': {
                    'breakout': lambda ind: ind['vol_breakout'] == True,
                    'news_catalyst': lambda ind: ind['has_news'] == True
                },
                'bypass_threshold': 1
            },
            # ... outros filtros
        }
    
    def validate(self, symbol: str, side: str, indicators: dict) -> Tuple[bool, str]:
        """
        Executa filtros em ordem de prioridade.
        Permite bypass estratégico para não perder setups válidos.
        """
        
        for filter_name, config in sorted(self.filters.items(), 
                                          key=lambda x: x[1]['priority']):
            
            # Executa filtro
            passed, reason = self._execute_filter(filter_name, symbol, side, indicators)
            
            # Se mandatório, falha imediata
            if not passed and config['mandatory']:
                return False, f"Filtro obrigatório falhou: {filter_name} - {reason}"
            
            # Se adaptativo, tenta bypass
            if not passed and not config['mandatory']:
                can_bypass = self._check_bypass(filter_name, indicators, config)
                
                if can_bypass:
                    logger.warning(
                        f"⚠️ {symbol}: Filtro '{filter_name}' BYPASSED | "
                        f"Razão: {reason} | "
                        f"Justificativa: Condições excepcionais detectadas"
                    )
                    continue  # Pula este filtro
                else:
                    return False, f"Filtro adaptativo falhou: {filter_name} - {reason}"
        
        return True, "Todos os filtros passaram"
    
    def _check_bypass(self, filter_name: str, indicators: dict, config: dict) -> bool:
        """
        Verifica se condições excepcionais justificam bypass.
        """
        conditions_met = 0
        
        for condition_name, condition_func in config['bypass_conditions'].items():
            if condition_func(indicators):
                conditions_met += 1
                logger.debug(f"  ✓ Condição bypass atendida: {condition_name}")
        
        return conditions_met >= config['bypass_threshold']
```

**Objetivo:** Aumentar taxa de execução de **2.9% para 15-20%** sem comprometer qualidade

---

### RF4: Proper ML Integration (Integração ML Profissional)

**Problemas atuais:**
1. Dados sintéticos no treinamento
2. Features inconsistentes (espera 16, recebe 8)
3. Threshold irreal (78%)
4. Nunca treina porque nunca opera

**Solução:**

```python
class MLTradingSystem:
    """
    Sistema ML profissional com retreino online e bootstrap inicial.
    """
    
    def __init__(self):
        self.predictor = MLSignalPredictor(confidence_threshold=0.55)  # Realista
        self.feature_engineer = FeatureEngineer()
        self.online_learner = OnlineLearner()
        
    def bootstrap_from_history(self):
        """
        FASE 1: Carrega dados históricos REAIS do MT5
        
        Processo:
        1. Busca últimos 90 dias de cada símbolo
        2. Calcula indicadores retroativamente
        3. Simula trades que TERIA feito
        4. Usa P&L real desses setups para treinar
        5. Mínimo: 500 exemplos por classe de ativo
        """
        
        for symbol in self.universe:
            # Baixa histórico
            df = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M15, 
                                     datetime.now() - timedelta(days=90),
                                     datetime.now())
            
            # Gera features para cada candle
            features_df = self.feature_engineer.compute_all(df)
            
            # Simula trades (walk-forward)
            simulated_trades = self._simulate_historical_trades(features_df)
            
            # Treina com resultados reais
            self.predictor.train(simulated_trades)
        
        logger.info(f"✅ Bootstrap completo: {len(simulated_trades)} trades sintéticos")
    
    def ensure_feature_consistency(self, raw_indicators: dict) -> np.ndarray:
        """
        Garante que features sempre tenham shape correto.
        
        Features padronizadas (16):
        [0] RSI (0-100)
        [1] ADX (0-100)
        [2] ATR% (0-10)
        [3] Volume Ratio (0-5)
        [4] Momentum (normalizado -1 a 1)
        [5] EMA Diff % (-5 a 5)
        [6] MACD (normalizado)
        [7] Price vs VWAP % (-5 a 5)
        [8-11] Fundamentals (P/E, ROE, Market Cap, Sector Score)
        [12] News Sentiment (-1 a 1)
        [13] Order Flow Imbalance (-1 a 1)
        [14] VIX BR (normalizado 0-1)
        [15] Book Imbalance (-1 a 1)
        """
        
        # Template com defaults
        features = np.zeros(16, dtype=np.float32)
        
        # Preenche com valores reais
        features[0] = raw_indicators.get('rsi', 50.0)
        features[1] = raw_indicators.get('adx', 20.0)
        # ... resto das 16 features
        
        # Valida range
        features = np.clip(features, -10, 10)  # Sanidade
        
        return features
    
    def predict_with_confidence_adjustment(self, symbol: str, 
                                          indicators: dict) -> dict:
        """
        Predição com threshold dinâmico baseado em contexto.
        """
        
        # Extrai features padronizadas
        features = self.ensure_feature_consistency(indicators)
        
        # Predição base
        prediction = self.predictor.predict(symbol, features)
        
        # Ajusta threshold baseado em:
        base_threshold = 0.55  # Inicial realista
        
        # 1. Volatilidade (VIX alto = threshold maior)
        vix = utils.get_vix_br()
        if vix > 30:
            base_threshold += 0.10  # 65%
        elif vix < 20:
            base_threshold -= 0.05  # 50%
        
        # 2. Win Rate recente do símbolo
        symbol_wr = self.get_recent_winrate(symbol, days=7)
        if symbol_wr > 0.65:
            base_threshold -= 0.08  # 47% (mais agressivo)
        elif symbol_wr < 0.40:
            base_threshold += 0.15  # 70% (mais defensivo)
        
        # 3. Hora do dia (mais conservador no início)
        hour = datetime.now().hour
        if 10 <= hour <= 11:  # Primeira hora ações
            base_threshold += 0.05
        elif 15 <= hour <= 16:  # Power hour
            base_threshold -= 0.05
        
        # Clamp final
        final_threshold = np.clip(base_threshold, 0.45, 0.75)
        
        # Aplica threshold ajustado
        prediction['threshold_used'] = final_threshold
        prediction['approved'] = prediction['confidence'] >= final_threshold
        
        return prediction
    
    def train_online(self, trade_result: dict):
        """
        Retreino incremental após cada trade.
        Não espera 100 trades - aprende continuamente.
        """
        
        # Adiciona ao buffer
        self.online_learner.add_sample(trade_result)
        
        # Treina se tiver mínimo de exemplos novos (20)
        if len(self.online_learner.buffer) >= 20:
            self.predictor.partial_fit(self.online_learner.buffer)
            self.online_learner.buffer.clear()
            
            logger.info("🔄 Retreino online executado (20 novos trades)")
```

---

### RF5: Concurrent Asset Processing (Processamento Concorrente)

**Problema atual:** Loop sequencial com latência de 12-15 segundos

**Solução:**

```python
class ConcurrentMarketScanner:
    """
    Processa múltiplos ativos em paralelo com priorização.
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.priority_queue = PriorityQueue()
        
    def scan_market(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Escaneia todos os símbolos simultaneamente.
        
        Priorização:
        1. FUTUROS (latência crítica) - Thread dedicada
        2. BLUE CHIPS (alta liquidez) - Pool compartilhado
        3. SMALL CAPS (baixa prioridade) - Pool compartilhado
        """
        
        # Separa por prioridade
        futures = [s for s in symbols if utils.is_future(s)]
        stocks = [s for s in symbols if not utils.is_future(s)]
        
        results = {}
        
        # Thread dedicada para futuros (baixa latência)
        if futures:
            future_results = self.executor.submit(
                self._scan_futures_fast, futures
            )
            results.update(future_results.result(timeout=3))  # Max 3s
        
        # Pool para ações (pode ser mais lento)
        if stocks:
            stock_futures = [
                self.executor.submit(self._analyze_symbol, s) 
                for s in stocks
            ]
            
            for future in as_completed(stock_futures, timeout=10):
                symbol, analysis = future.result()
                results[symbol] = analysis
        
        return results
    
    def _scan_futures_fast(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Processamento otimizado para futuros.
        
        Otimizações:
        - Cache agressivo (5s)
        - Menos indicadores (apenas críticos)
        - Prioriza velocidade sobre precisão
        """
        
        results = {}
        
        for symbol in symbols:
            # Check cache primeiro
            cached = self.cache.get(symbol, max_age=5)
            if cached:
                results[symbol] = cached
                continue
            
            # Indicadores mínimos (rápido)
            df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 50)
            
            indicators = {
                'rsi': utils.get_rsi(df, period=14),
                'ema_fast': df['close'].ewm(span=9).mean().iloc[-1],
                'ema_slow': df['close'].ewm(span=21).mean().iloc[-1],
                'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1],
                'price': df['close'].iloc[-1]
            }
            
            # Cache para próxima iteração
            self.cache.set(symbol, indicators, ttl=5)
            
            results[symbol] = indicators
        
        return results
```

**Resultado esperado:**
- Latência FUTUROS: **12s → 2s** (redução de 83%)
- Throughput geral: **5 símbolos/minuto → 30 símbolos/minuto**

---

### RF6: State Management & Persistence (Gestão de Estado Robusta)

**Problema atual:** Estado diário frágil, sem validação, com race conditions

**Solução:**

```python
class StateManager:
    """
    Gerenciamento robusto de estado com ACID guarantees.
    """
    
    def __init__(self, db_path: str = "trading_state.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.RLock()
        self._init_schema()
        
    def _init_schema(self):
        """
        Cria tabelas com constraints.
        """
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                trading_date DATE PRIMARY KEY,
                equity_start REAL NOT NULL,
                equity_max REAL NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins_count INTEGER DEFAULT 0,
                loss_streak INTEGER DEFAULT 0,
                circuit_breaker_active BOOLEAN DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CHECK (equity_start > 0),
                CHECK (equity_max >= equity_start),
                CHECK (trades_count >= 0),
                CHECK (wins_count >= 0),
                CHECK (loss_streak >= 0)
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS symbol_limits (
                symbol TEXT NOT NULL,
                trading_date DATE NOT NULL,
                trades_count INTEGER DEFAULT 0,
                losses_count INTEGER DEFAULT 0,
                last_sl_time TIMESTAMP,
                cooldown_until TIMESTAMP,
                
                PRIMARY KEY (symbol, trading_date),
                CHECK (trades_count >= losses_count)
            )
        """)
        
        self.db.commit()
    
    def save_state_atomic(self, state: dict):
        """
        Salva estado com transação ACID.
        Garante que não haverá corrupção mesmo em crash.
        """
        
        with self.lock:
            try:
                self.db.execute("BEGIN TRANSACTION")
                
                # Atualiza/insere estado
                self.db.execute("""
                    INSERT OR REPLACE INTO daily_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state['trading_date'],
                    state['equity_start'],
                    state['equity_max'],
                    state['trades_count'],
                    state['wins_count'],
                    state['loss_streak'],
                    state['circuit_breaker_active'],
                    datetime.now()
                ))
                
                self.db.execute("COMMIT")
                logger.debug("💾 Estado salvo com sucesso (transacional)")
                
            except Exception as e:
                self.db.execute("ROLLBACK")
                logger.error(f"❌ Erro ao salvar estado: {e}")
                raise
    
    def load_state_with_validation(self, date: date) -> Optional[dict]:
        """
        Carrega estado com validação de integridade.
        """
        
        with self.lock:
            cursor = self.db.execute("""
                SELECT * FROM daily_state WHERE trading_date = ?
            """, (date,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            state = dict(zip([col[0] for col in cursor.description], row))
            
            # Validações de sanidade
            if state['equity_start'] <= 0:
                logger.error("⚠️ Estado corrompido: equity_start inválido")
                return None
            
            if state['equity_max'] < state['equity_start']:
                logger.warning("⚠️ Estado suspeito: equity_max < equity_start (corrigindo)")
                state['equity_max'] = state['equity_start']
            
            return state
    
    def reset_daily_if_needed(self):
        """
        Reset automático e seguro na virada do dia.
        """
        
        today = datetime.now().date()
        
        with self.lock:
            # Verifica se já tem entrada de hoje
            cursor = self.db.execute("""
                SELECT COUNT(*) FROM daily_state WHERE trading_date = ?
            """, (today,))
            
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Novo dia - cria entrada limpa
                account = mt5.account_info()
                
                if account:
                    self.db.execute("""
                        INSERT INTO daily_state VALUES (?, ?, ?, 0, 0, 0, 0, ?)
                    """, (today, account.equity, account.equity, datetime.now()))
                    
                    # Limpa limites de símbolos (novo dia)
                    self.db.execute("""
                        DELETE FROM symbol_limits WHERE trading_date < ?
                    """, (today - timedelta(days=7),))  # Mantém 7 dias
                    
                    self.db.commit()
                    
                    logger.info(f"🌅 Novo dia iniciado: {today} | Equity: R$ {account.equity:,.2f}")
```

---

### RF7: Advanced Position Management (Gestão Avançada de Posições)

**Problema atual:** Pyramid mal feito, trailing stop não funciona, SL não ajusta

**Solução:**

```python
class PositionManager:
    """
    Gerencia posições com lógica sofisticada de escalonamento.
    """
    
    def __init__(self):
        self.positions = {}  # {ticket: PositionState}
        self.lock = threading.RLock()
        
    def add_pyramid_leg(self, symbol: str, existing_ticket: int, 
                       side: str, indicators: dict) -> Optional[dict]:
        """
        Adiciona perna de pirâmide com validações rigorosas.
        
        Condições obrigatórias:
        1. Lucro atual >= 1.5 ATR
        2. Nova entrada >= 0.8 ATR de distância
        3. Direção da tendência mantida
        4. ADX ainda crescente
        5. Volume ainda alto
        """
        
        with self.lock:
            # Valida posição existente
            existing_pos = mt5.positions_get(ticket=existing_ticket)[0]
            
            current_price = mt5.symbol_info_tick(symbol).bid if side == "BUY" else .ask
            entry_price = existing_pos.price_open
            
            # 1. Check lucro mínimo
            profit_distance = abs(current_price - entry_price)
            atr = indicators['atr']
            
            if profit_distance < 1.5 * atr:
                return None, "Lucro insuficiente para pirâmide"
            
            # 2. Check distância mínima (evita entrada muito próxima)
            min_distance = 0.8 * atr
            if profit_distance < min_distance:
                return None, f"Distância muito pequena ({profit_distance:.2f} < {min_distance:.2f})"
            
            # 3. Valida que tendência continua
            if side == "BUY" and indicators['ema_diff'] <= 0:
                return None, "Tendência revertida (EMA negativa)"
            
            if side == "SELL" and indicators['ema_diff'] >= 0:
                return None, "Tendência revertida (EMA positiva)"
            
            # 4. ADX deve estar crescente
            adx_series = utils.get_adx_series(indicators['df'])
            if adx_series.iloc[-1] <= adx_series.iloc[-2]:
                return None, "ADX em queda (força reduzindo)"
            
            # 5. Volume ainda alto
            if indicators['volume_ratio'] < 1.2:
                return None, "Volume não sustenta pirâmide"
            
            # ✅ Todas as validações passaram - calcula nova posição
            new_volume = existing_pos.volume * 0.50  # 50% da posição original
            new_sl = current_price - (1.5 * atr) if side == "BUY" else current_price + (1.5 * atr)
            
            # IMPORTANTE: Ajusta SL da posição original
            self._adjust_original_sl_after_pyramid(existing_ticket, new_sl)
            
            return {
                'volume': new_volume,
                'sl': new_sl,
                'tp': existing_pos.tp,  # Mantém TP original
                'approved': True
            }
    
    def _adjust_original_sl_after_pyramid(self, ticket: int, new_sl: float):
        """
        Move SL da primeira perna para breakeven + comissões.
        Protege lucro já conquistado.
        """
        
        pos = mt5.positions_get(ticket=ticket)[0]
        
        # SL no breakeven (considera custos)
        costs_pct = 0.0015  # ~0.15% (comissões + slippage)
        breakeven = pos.price_open * (1 + costs_pct if pos.type == mt5.POSITION_TYPE_BUY else 1 - costs_pct)
        
        # Só move para cima (compra) ou para baixo (venda)
        safe_sl = max(breakeven, new_sl) if pos.type == mt5.POSITION_TYPE_BUY else min(breakeven, new_sl)
        
        request = {
            'action': mt5.TRADE_ACTION_SLTP,
            'position': ticket,
            'sl': safe_sl,
            'tp': pos.tp
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"✅ Pyramid: SL ajustado para breakeven+custos | "
                f"Ticket {ticket} | Novo SL: {safe_sl:.2f}"
            )
        else:
            logger.error(f"❌ Falha ao ajustar SL após pirâmide: {result.comment}")
    
    def apply_dynamic_trailing_stop(self, symbol: str, ticket: int):
        """
        Trailing stop que se adapta à volatilidade.
        
        Lógica:
        - ATR alto (>3%) → trailing largo (2.5 ATR)
        - ATR médio (1-3%) → trailing normal (2.0 ATR)
        - ATR baixo (<1%) → trailing apertado (1.5 ATR)
        
        Proteção contra whipsaw:
        - Só move SL para melhor, nunca para pior
        - Mínimo de 10 minutos desde última mudança
        - Não move se preço volátil (última vela > 2 ATR)
        """
        
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return
        
        pos = pos[0]
        side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        
        # Calcula ATR
        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 50)
        atr = utils.get_atr(df)
        atr_pct = (atr / pos.price_open) * 100
        
        # Define multiplicador baseado em volatilidade
        if atr_pct > 3.0:
            multiplier = 2.5  # Largo
        elif atr_pct > 1.0:
            multiplier = 2.0  # Normal
        else:
            multiplier = 1.5  # Apertado
        
        # Calcula novo SL
        current_price = mt5.symbol_info_tick(symbol).bid if side == "BUY" else .ask
        
        if side == "BUY":
            new_sl = current_price - (multiplier * atr)
            # Só move para cima
            if new_sl > pos.sl:
                self._move_sl(ticket, new_sl)
        else:
            new_sl = current_price + (multiplier * atr)
            # Só move para baixo
            if new_sl < pos.sl:
                self._move_sl(ticket, new_sl)
```

---

## 🎯 MÉTRICAS DE SUCESSO

Após implementação completa, o sistema deve atingir:

| Métrica | Atual | Meta | Criticidade |
|---------|-------|------|-------------|
| Win Rate | 28% | 58%+ | 🔴 Crítico |
| Taxa de Execução | 2.9% | 18%+ | 🔴 Crítico |
| Latência (Futuros) | 12s | <3s | 🔴 Crítico |
| Utilizção Capital | 45% | 85%+ | 🟠 Alto |
| Drawdown Máximo | 18% | <8% | 🔴 Crítico |
| Profit Factor | 0.75 | 1.8+ | 🔴 Crítico |
| Posições Órfãs | 15% | <2% | 🟠 Alto |
| Memory Leaks | Sim (4GB/8h) | <500MB/8h | 🟡 Médio |
| Uptime | 75% | 99%+ | 🟠 Alto |

---

## 📐 ARQUITETURA PROPOSTA

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN ORCHESTRATOR                        │
│  (Thread-safe, com circuit breakers e health monitoring)   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   FUTURES    │    │    STOCKS    │    │   COMMON     │
│   PROCESSOR  │    │  PROCESSOR   │    │   SERVICES   │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ - M5/M15 TF  │    │ - M15/H1 TF  │    │ - State Mgr  │
│ - 09:00-17:50│    │ - 10:00-17:55│    │ - ML Engine  │
│ - Margem Calc│    │ - Linear Calc│    │ - Risk Mgr   │
│ - Points SL  │    │ - % SL       │    │ - Telegram   │
│ - 35% Capital│    │ - 65% Capital│    │ - Logging    │
│ - Latency <3s│    │ - Cache 10s  │    │ - Health Mon │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   MT5 INTERFACE  │
                    │  (Thread-locked) │
                    └──────────────────┘
```

---

## 🔄 FLUXO DE EXECUÇÃO CORRIGIDO

```python
def main_trading_loop():
    """
    Loop principal refatorado - multi-asset simultâneo.
    """
    
    # 1. Inicialização
    asset_manager = AssetClassManager()
    mtf_engine = MultiTimeframeEngine()
    ml_system = MLTradingSystem()
    filter_chain = FilterChain()
    state_manager = StateManager()
    position_manager = PositionManager()
    market_scanner = ConcurrentMarketScanner(max_workers=4)
    
    # 2. Bootstrap ML (primeira vez)
    if ml_system.needs_bootstrap():
        ml_system.bootstrap_from_history()
    
    while True:
        try:
            # 3. Reset diário se necessário
            state_manager.reset_daily_if_needed()
            
            # 4. Escaneia mercado (paralelo)
            all_indicators = market_scanner.scan_market(universe_symbols)
            
            # 5. Processa FUTUROS (prioridade alta)
            for symbol in futures_symbols:
                indicators = all_indicators.get(symbol)
                if not indicators:
                    continue
                
                # Determina sinal
                side = "BUY" if indicators['ema_fast'] > indicators['ema_slow'] else "SELL"
                
                # Validação completa
                if not self._validate_entry_complete(symbol, side, indicators,
                                                     asset_manager, mtf_engine, 
                                                     ml_system, filter_chain):
                    continue
                
                # Executa (fast path)
                self._execute_trade(symbol, side, indicators, asset_manager, position_manager)
            
            # 6. Processa AÇÕES (prioridade normal)
            for symbol in stock_symbols:
                indicators = all_indicators.get(symbol)
                if not indicators:
                    continue
                
                # Mesma validação (mas timeframes diferentes)
                side = "BUY" if indicators['ema_fast'] > indicators['ema_slow'] else "SELL"
                
                if not self._validate_entry_complete(symbol, side, indicators,
                                                     asset_manager, mtf_engine, 
                                                     ml_system, filter_chain):
                    continue
                
                # Executa
                self._execute_trade(symbol, side, indicators, asset_manager, position_manager)
            
            # 7. Gestão de posições (paralelo por ticket)
            with ThreadPoolExecutor(max_workers=8) as executor:
                positions = mt5.positions_get()
                
                for pos in positions:
                    executor.submit(
                        position_manager.manage_position,
                        pos.ticket, pos.symbol, all_indicators.get(pos.symbol)
                    )
            
            # 8. Salva estado
            state_manager.save_state_atomic(current_state)
            
            # 9. Sleep inteligente (1s futuros ativos, 5s caso contrário)
            if futures_have_positions():
                time.sleep(1)
            else:
                time.sleep(5)
            
        except Exception as e:
            logger.error(f"Erro no loop principal: {e}", exc_info=True)
            time.sleep(10)

def _validate_entry_complete(self, symbol: str, side: str, indicators: dict,
                             asset_manager, mtf_engine, ml_system, filter_chain) -> bool:
    """
    Validação unificada com fallback inteligente.
    """
    
    # 1. Capital disponível (obrigatório)
    config = asset_manager.get_config(symbol)
    if not self._check_capital(symbol, config):
        return False
    
    # 2. MTF (obrigatório)
    mtf_ok, reason = mtf_engine.validate_entry(symbol, side, config.timeframe)
    if not mtf_ok:
        log_rejection(symbol, "MTF", reason)
        return False
    
    # 3. ML (adaptativo)
    ml_result = ml_system.predict_with_confidence_adjustment(symbol, indicators)
    
    if not ml_result['approved']:
        # Permite bypass se sinais técnicos muito fortes
        if indicators['adx'] > 45 and indicators['volume_ratio'] > 2.5:
            logger.warning(f"⚠️ {symbol}: ML rejeitou mas setup excepcional - BYPASS")
        else:
            log_rejection(symbol, "ML", f"Conf {ml_result['confidence']:.0%} < {ml_result['threshold_used']:.0%}")
            return False
    
    # 4. Filter Chain (outros filtros)
    passed, reason = filter_chain.validate(symbol, side, indicators)
    if not passed:
        log_rejection(symbol, "FilterChain", reason)
        return False
    
    # ✅ Todas as validações passaram
    return True
```

---

## 🚀 PLANO DE IMPLEMENTAÇÃO

### Fase 1: Fundação (Semana 1)
1. ✅ Criar `AssetClassManager`
2. ✅ Refatorar `calculate_position_size` por classe
3. ✅ Implementar `StateManager` com SQLite
4. ✅ Testes unitários para cada componente

**Validação:** Position sizing diferenciado funciona, estado persiste entre restarts

---

### Fase 2: Multi-Timeframe (Semana 2)
1. ✅ Implementar `MultiTimeframeEngine`
2. ✅ Integrar validação MTF no fluxo de entrada
3. ✅ Adicionar logs detalhados de MTF
4. ✅ Backtesting em dados históricos

**Validação:** Taxa de falsos sinais reduz de 40% para <15%

---

### Fase 3: ML Profissional (Semana 3)
1. ✅ Bootstrap a partir de histórico MT5
2. ✅ Refatorar features para 16 consistentes
3. ✅ Implementar online learning
4. ✅ Ajustar thresholds para 0.55 base

**Validação:** Modelo começa a operar desde o dia 1, retreina continuamente

---

### Fase 4: Processamento Paralelo (Semana 4)
1. ✅ Implementar `ConcurrentMarketScanner`
2. ✅ Thread dedicada para futuros
3. ✅ Pool compartilhado para ações
4. ✅ Profiling e otimização

**Validação:** Latência futuros <3s, throughput 6x maior

---

### Fase 5: Gestão Avançada (Semana 5)
1. ✅ Pyramid com validações rigorosas
2. ✅ Trailing stop dinâmico
3. ✅ Ajuste de SL após pyramid
4. ✅ Tests stress em paper trading

**Validação:** Pyramid com 60%+ win rate, trailing não gera whipsaw

---

### Fase 6: Filtros Inteligentes (Semana 6)
1. ✅ Refatorar em `FilterChain`
2. ✅ Implementar bypass conditions
3. ✅ Logging detalhado de rejeições
4. ✅ A/B testing de configurações

**Validação:** Taxa de execução sobe de 2.9% para 15-20%

---

### Fase 7: Monitoramento & Deploy (Semana 7)
1. ✅ Dashboard real-time (Streamlit)
2. ✅ Alertas Telegram críticos
3. ✅ Health monitoring automático
4. ✅ Deploy gradual (paper → 20% capital → 100%)

**Validação:** Uptime 99%+, sem memory leaks, todas as métricas atingidas

---

## 📚 RECURSOS NECESSÁRIOS

### Bibliotecas Adicionais
```bash
pip install sqlite3  # State management
pip install asyncio  # Async operations
pip install aiohttp  # Async HTTP
pip install prometheus-client  # Metrics
```

### Hardware Recomendado
- **CPU:** 8 cores+ (processamento paralelo)
- **RAM:** 16GB+ (evita swapping)
- **SSD:** NVMe (I/O rápido para logs e cache)
- **Network:** Fibra/baixa latência (futuros são sensíveis)

---

## ⚠️ AVISOS CRÍTICOS

1. **NÃO implemente tudo de uma vez** - Risco de criar mais bugs
2. **SEMPRE teste em paper trading** antes de cada fase
3. **Mantenha rollback fácil** - Git branches por fase
4. **Monitore memory/CPU** durante toda implementação
5. **Documente cada mudança** - Facilita debug futuro

---

## 🎓 CONHECIMENTOS APLICADOS

Esta solução baseia-se em:
- **Design Patterns:** Strategy, Factory, Observer, State Machine
- **Concurrent Programming:** Thread pools, locks, semaphores, priority queues
- **Financial Engineering:** Kelly Criterion, multi-asset allocation, regime detection
- **Machine Learning:** Online learning, feature engineering, ensemble methods
- **Systems Architecture:** Microservices mindset, separation of concerns

---

**Última Atualização:** 28/01/2026  
**Versão do Prompt:** 1.0  
**Complexidade Estimada:** Alta (7 semanas de desenvolvimento sênior)  
**ROI Esperado:** 300-400% (de -R$ 15k/mês para +R$ 50k/mês)
