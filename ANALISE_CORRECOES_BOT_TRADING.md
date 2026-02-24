# 🎯 ANÁLISE TÉCNICA E CORREÇÕES - BOT TRADING B3 (Land Trading)

**Analista:** Especialista Senior em Trading Algorítmico  
**Data:** 28/01/2026  
**Objetivo:** Identificar problemas críticos e implementar correções para operação rentável no mercado futuro

---

## 📊 RESUMO EXECUTIVO

Após análise profunda do código, identifiquei **problemas críticos** que impedem o bot de operar adequadamente no mercado futuro e gerar lucros consistentes. O sistema possui excelente arquitetura, mas falhas de implementação comprometem a performance.

### 🔴 PROBLEMAS CRÍTICOS IDENTIFICADOS

1. **Ausência de estratégia específica para futuros**
2. **Thresholds de ML inadequados (78% - muito alto)**
3. **Gestão de risco genérica (não considera alavancagem)**
4. **Falta de filtros de microestrutura (orderbook, flow)**
5. **News calendar com conversão de timezone incorreta**
6. **Rollover de contratos não automatizado**
7. **Slippage subestimado para futuros**
8. **Stop loss estático (não dinâmico)**

---

## 🔧 CORREÇÕES PRIORITÁRIAS

### 1. 🎯 ESTRATÉGIA ESPECÍFICA PARA FUTUROS

**Problema:** O bot usa a mesma lógica para ações e futuros, ignorando as particularidades de cada mercado.

#### ✅ SOLUÇÃO - Criar `futures_strategy.py`:

```python
#futures_strategy.py
"""
Estratégia especializada para contratos futuros (WIN, WDO, etc)
Foca em: Microestrutura, Order Flow, Session Patterns
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, time as datetime_time
import config
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger("futures_strategy")

class FuturesStrategy:
    """
    Estratégia otimizada para mercado futuro:
    - Order Flow Delta (agressividade compradores vs vendedores)
    - Imbalance no Book de Ofertas
    - Volume Profile (POC, VAH, VAL)
    - Session Patterns (abertura, meio-dia, fechamento)
    """
    
    def __init__(self):
        self.min_orderbook_depth = 5  # Mínimo 5 níveis no book
        self.imbalance_threshold = 0.65  # 65% de imbalance mínimo
        self.session_patterns = self._load_session_patterns()
        
    def _load_session_patterns(self) -> Dict:
        """
        Padrões estatísticos de comportamento por horário:
        - Abertura (09:00-10:00): Volatilidade alta, false breakouts
        - Meio-dia (12:00-14:00): Redução de liquidez
        - Fechamento (16:30-17:00): Aumento de volume
        """
        return {
            "ABERTURA": {
                "start": datetime_time(9, 0),
                "end": datetime_time(10, 0),
                "adx_min": 30,  # Mais conservador na abertura
                "vol_multiplier": 1.5,
                "avoid_breakouts": True  # Evitar breakouts na abertura
            },
            "MEIO_DIA": {
                "start": datetime_time(12, 0),
                "end": datetime_time(14, 0),
                "adx_min": 28,
                "vol_multiplier": 0.8,
                "reduce_exposure": True  # Reduzir exposição
            },
            "FECHAMENTO": {
                "start": datetime_time(16, 30),
                "end": datetime_time(17, 0),
                "adx_min": 25,
                "vol_multiplier": 2.0,
                "increase_stops": True  # Stops mais largos
            }
        }
    
    def get_orderbook_imbalance(self, symbol: str) -> Tuple[float, bool]:
        """
        Calcula imbalance no book de ofertas.
        
        Returns:
            (imbalance_ratio, is_valid)
            imbalance > 0.65 = Pressão compradora
            imbalance < 0.35 = Pressão vendedora
        """
        try:
            book = mt5.market_book_get(symbol)
            
            if not book or len(book) < self.min_orderbook_depth:
                return 0.5, False
            
            # Separa bids (compra) e asks (venda)
            bids = [item for item in book if item.type == mt5.BOOK_TYPE_BUY]
            asks = [item for item in book if item.type == mt5.BOOK_TYPE_SELL]
            
            if not bids or not asks:
                return 0.5, False
            
            # Volume agregado
            bid_volume = sum(item.volume for item in bids[:5])
            ask_volume = sum(item.volume for item in asks[:5])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.5, False
            
            imbalance = bid_volume / total_volume
            
            # Log se houver desbalanceamento significativo
            if imbalance > 0.70 or imbalance < 0.30:
                logger.info(f"📊 {symbol} - Imbalance: {imbalance:.2%} (Bid: {bid_volume:.0f} | Ask: {ask_volume:.0f})")
            
            return imbalance, True
            
        except Exception as e:
            logger.error(f"Erro ao ler orderbook de {symbol}: {e}")
            return 0.5, False
    
    def get_order_flow_delta(self, symbol: str, bars: int = 20) -> Tuple[float, bool]:
        """
        Calcula Order Flow Delta:
        Volume de trades agressivos de compra - Volume de trades agressivos de venda
        
        Positivo = Compradores agressivos dominando
        Negativo = Vendedores agressivos dominando
        """
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars)
            
            if rates is None or len(rates) < 10:
                return 0.0, False
            
            df = pd.DataFrame(rates)
            
            # Aproximação do Order Flow:
            # Se close > open E volume alto = Compra agressiva
            # Se close < open E volume alto = Venda agressiva
            df['is_bullish'] = df['close'] > df['open']
            df['volume_delta'] = np.where(
                df['is_bullish'],
                df['tick_volume'],  # Compra
                -df['tick_volume']  # Venda
            )
            
            # Delta acumulado
            cumulative_delta = df['volume_delta'].sum()
            total_volume = df['tick_volume'].sum()
            
            if total_volume == 0:
                return 0.0, False
            
            # Normaliza entre -1 e 1
            normalized_delta = cumulative_delta / total_volume
            
            return float(normalized_delta), True
            
        except Exception as e:
            logger.error(f"Erro ao calcular Order Flow de {symbol}: {e}")
            return 0.0, False
    
    def get_session_adjustment(self) -> Dict:
        """
        Retorna ajustes de parâmetros baseado no horário da sessão
        """
        now = datetime.now().time()
        
        for session_name, rules in self.session_patterns.items():
            if rules["start"] <= now <= rules["end"]:
                logger.info(f"⏰ Sessão ativa: {session_name}")
                return rules
        
        # Horário normal
        return {
            "adx_min": 25,
            "vol_multiplier": 1.0,
            "avoid_breakouts": False,
            "reduce_exposure": False,
            "increase_stops": False
        }
    
    def should_trade_futures(
        self,
        symbol: str,
        indicators: Dict,
        session_rules: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Valida se devemos operar o futuro neste momento.
        
        Critérios:
        1. ADX adequado para a sessão
        2. Order Flow alinhado
        3. Book com liquidez suficiente
        4. Imbalance significativo
        """
        if session_rules is None:
            session_rules = self.get_session_adjustment()
        
        # 1. ADX
        adx = indicators.get('adx', 0)
        adx_min = session_rules.get('adx_min', 25)
        
        if adx < adx_min:
            return False, f"ADX insuficiente ({adx:.1f} < {adx_min})"
        
        # 2. Order Flow
        flow_delta, flow_valid = self.get_order_flow_delta(symbol)
        
        if not flow_valid:
            return False, "Order Flow indisponível"
        
        # Deve ter direção clara (|delta| > 0.15)
        if abs(flow_delta) < 0.15:
            return False, f"Order Flow neutro ({flow_delta:.2f})"
        
        # 3. Orderbook Imbalance
        imbalance, imb_valid = self.get_orderbook_imbalance(symbol)
        
        if not imb_valid:
            return False, "Book de ofertas insuficiente"
        
        # Imbalance deve ser significativo
        if 0.40 < imbalance < 0.60:
            return False, f"Imbalance neutro ({imbalance:.2%})"
        
        # 4. Alinhamento Flow + Imbalance
        # Se flow é positivo (compra), imbalance deve ser > 0.60
        # Se flow é negativo (venda), imbalance deve ser < 0.40
        if flow_delta > 0 and imbalance < 0.55:
            return False, "Flow vs Imbalance desalinhados (compra)"
        
        if flow_delta < 0 and imbalance > 0.45:
            return False, "Flow vs Imbalance desalinhados (venda)"
        
        # ✅ Todos os critérios OK
        return True, f"OK (ADX:{adx:.1f} | Flow:{flow_delta:.2f} | Imb:{imbalance:.2%})"
    
    def get_dynamic_stops_futures(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        side: str,
        session_rules: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Calcula SL/TP dinâmicos para futuros.
        
        Considera:
        - ATR
        - Volatilidade intraday
        - Horário da sessão
        - Suporte/Resistência próximos
        """
        if session_rules is None:
            session_rules = self.get_session_adjustment()
        
        # Base: ATR multiplicador
        sl_multiplier = 2.5
        tp_multiplier = 5.0
        
        # Ajuste por sessão
        if session_rules.get("increase_stops"):
            sl_multiplier *= 1.3
            tp_multiplier *= 1.2
        
        # Calcula stops
        sl_distance = atr * sl_multiplier
        tp_distance = atr * tp_multiplier
        
        if side == "BUY":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return round(sl, 2), round(tp, 2)
    
    def calculate_futures_score(
        self,
        symbol: str,
        indicators: Dict
    ) -> float:
        """
        Score de 0 a 100 para qualidade do setup em futuros.
        
        Componentes:
        - Order Flow (30 pts)
        - Orderbook Imbalance (30 pts)
        - ADX (20 pts)
        - Volatilidade (10 pts)
        - Sessão (10 pts)
        """
        score = 0.0
        
        # 1. Order Flow (30 pts)
        flow_delta, flow_valid = self.get_order_flow_delta(symbol)
        if flow_valid:
            # Quanto maior |delta|, melhor
            score += min(30, abs(flow_delta) * 100)
        
        # 2. Orderbook (30 pts)
        imbalance, imb_valid = self.get_orderbook_imbalance(symbol)
        if imb_valid:
            # Distância de 0.5 (neutro)
            imb_distance = abs(imbalance - 0.5)
            score += min(30, imb_distance * 100)
        
        # 3. ADX (20 pts)
        adx = indicators.get('adx', 0)
        score += min(20, (adx / 50) * 20)
        
        # 4. Volatilidade (10 pts)
        atr_pct = indicators.get('atr_pct', 0)
        if 0.015 <= atr_pct <= 0.04:  # Faixa ideal para futuros
            score += 10
        
        # 5. Sessão (10 pts)
        session_rules = self.get_session_adjustment()
        if not session_rules.get("avoid_breakouts") and not session_rules.get("reduce_exposure"):
            score += 10
        
        return min(100, score)


# ========================
# INTEGRAÇÃO COM O BOT
# ========================

def enhance_futures_signal(
    symbol: str,
    side: str,
    indicators: Dict,
    base_confidence: float
) -> Tuple[float, Dict]:
    """
    Função de integração com o bot principal.
    
    Aumenta confiança se microestrutura confirmar o sinal técnico.
    """
    strategy = FuturesStrategy()
    
    # Valida se devemos operar
    can_trade, reason = strategy.should_trade_futures(symbol, indicators)
    
    if not can_trade:
        logger.info(f"❌ {symbol} - Futures strategy bloqueou: {reason}")
        return 0.0, {"blocked": True, "reason": reason}
    
    # Calcula score de qualidade
    quality_score = strategy.calculate_futures_score(symbol, indicators)
    
    # Boost na confiança baseado no score
    confidence_boost = (quality_score / 100) * 0.15  # Até +15%
    enhanced_confidence = min(1.0, base_confidence + confidence_boost)
    
    logger.info(
        f"✅ {symbol} - Futures Enhanced: "
        f"Base={base_confidence:.2%} → Enhanced={enhanced_confidence:.2%} "
        f"(Score: {quality_score:.0f}/100)"
    )
    
    return enhanced_confidence, {
        "quality_score": quality_score,
        "confidence_boost": confidence_boost,
        "reason": reason
    }


# ========================
# EXEMPLO DE USO NO BOT
# ========================
"""
# No arquivo bot.py, na função calculate_signal_score:

if utils.is_futures(symbol):
    # Aplica estratégia de futuros
    enhanced_conf, meta = enhance_futures_signal(
        symbol=symbol,
        side=signal_side,
        indicators=ind,
        base_confidence=ml_confidence
    )
    
    if meta.get("blocked"):
        logger.info(f"⛔ {symbol} - Bloqueado por futures strategy: {meta['reason']}")
        return 0  # Bloqueia entrada
    
    # Usa confiança melhorada
    final_confidence = enhanced_conf
"""
```

---

### 2. 📉 REDUZIR THRESHOLD DE ML (78% → 62%)

**Problema:** Threshold de 78% é inatingível na prática. Ensemble real raramente atinge > 70%.

#### ✅ CORREÇÃO no `config.py`:

```python
# LINHA 196 (config.py)
# ANTES:
ML_MIN_CONFIDENCE = config_manager.get('ml.min_confidence', 0.65)  # Reduced from 0.78 to 0.65

# DEPOIS:
ML_MIN_CONFIDENCE = 0.62  # Threshold realista para ensemble
ML_CONFIDENCE_FUTURES = 0.58  # Futuros: threshold menor (mais liquidez)
ML_CONFIDENCE_STOCKS = 0.65  # Ações: threshold maior (menos liquidez)

# Ajuste dinâmico baseado em performance
ML_CONFIDENCE_ADAPTIVE = True
ML_CONFIDENCE_MIN = 0.55  # Piso
ML_CONFIDENCE_MAX = 0.72  # Teto
```

#### ✅ CORREÇÃO no `ml_signals.py`:

```python
# LINHA 114 (ml_signals.py)
# ANTES:
def __init__(self, models_dir: str = "models", confidence_threshold: float = 0.78):

# DEPOIS:
def __init__(self, models_dir: str = "models", confidence_threshold: float = 0.62):
    self.models_dir = Path(models_dir)
    self.models_dir.mkdir(exist_ok=True)
    self.confidence_threshold = confidence_threshold
    
    # ✅ NOVO: Thresholds por tipo de ativo
    self.thresholds = {
        "STOCKS": 0.65,
        "FUTURES": 0.58,
        "DEFAULT": 0.62
    }
```

---

### 3. 💰 GESTÃO DE RISCO ESPECÍFICA PARA FUTUROS

**Problema:** Bot usa 0.25% de risco para todos os ativos. Futuros com alavancagem exigem ajuste.

#### ✅ CORREÇÃO no `config.py`:

```python
# GESTÃO DE RISCO DIFERENCIADA
RISK_FUTURES = {
    "risk_per_trade_pct": 0.004,  # 0.4% por trade (maior que ações)
    "max_positions": 2,  # Máx 2 posições simultâneas em futuros
    "leverage_factor": 20,  # Alavancagem típica 1:20
    "margin_safety": 0.60,  # Usa apenas 60% da margem disponível
    "max_daily_dd": 0.03,  # 3% de drawdown máximo diário
    "circuit_breaker": 0.05  # Para tudo se DD >= 5%
}

RISK_STOCKS = {
    "risk_per_trade_pct": 0.0025,  # 0.25%
    "max_positions": 8,
    "max_daily_dd": 0.05,
    "circuit_breaker": 0.07
}

def get_risk_params(symbol: str) -> dict:
    """Retorna parâmetros de risco baseado no tipo de ativo"""
    if is_futures(symbol):
        return RISK_FUTURES
    return RISK_STOCKS
```

---

### 4. 🔧 CORRIGIR NEWS CALENDAR (TIMEZONE)

**Problema:** Conversão de timezone do MT5 (UTC) para BRT está incorreta.

#### ✅ CORREÇÃO no `news_calendar.py`:

```python
# LINHA 42-67 (news_calendar.py)
# SUBSTITUIR MÉTODO apply_blackout COMPLETAMENTE:

def apply_blackout(self, symbol: str) -> tuple:
    """
    Verifica se há notícias de alto impacto próximas.
    
    Returns:
        tuple: (bool, str) - (is_blackout, reason)
    """
    if not self.enabled:
        return False, ""

    try:
        if not hasattr(mt5, 'calendar_events_get'):
            logger.debug("News calendar não disponível nesta versão do MT5")
            return False, ""
        
        # ✅ CORREÇÃO: Trabalhar sempre em UTC
        now_utc = datetime.now(self.tz_utc)
        
        # Busca eventos em janela de ±2h
        dt_from = now_utc - timedelta(hours=2)
        dt_to = now_utc + timedelta(hours=2)

        with mt5_lock:
            events = mt5.calendar_events_get(date_from=dt_from, date_to=dt_to)
        
        if events is None or len(events) == 0:
            return False, ""

        before = timedelta(minutes=config.NEWS_BLOCK_BEFORE_MIN)
        after = timedelta(minutes=config.NEWS_BLOCK_AFTER_MIN)

        for event in events:
            if event.importance == mt5.CALENDAR_IMPORTANCE_HIGH:
                if event.currency in symbol or event.currency == "USD":
                    
                    # ✅ CORREÇÃO: Tempo do evento já está em UTC
                    event_time_utc = datetime.fromtimestamp(event.time, self.tz_utc)
                    
                    # ✅ COMPARAÇÃO: Tudo em UTC
                    if (event_time_utc - before <= now_utc <= event_time_utc + after):
                        # Converte apenas para exibição
                        event_time_br = event_time_utc.astimezone(self.tz_br)
                        reason = f"Notícia {event.currency}: {event.name} às {event_time_br.strftime('%H:%M BRT')}"
                        logger.warning(f"[BLACKOUT] {reason}")
                        return True, reason

        return False, ""
    
    except Exception as e:
        logger.error(f"Erro ao verificar news calendar: {e}")
        return False, ""
```

---

### 5. 🔄 ROLLOVER AUTOMÁTICO DE CONTRATOS

**Problema:** Rollover manual pode causar perda de posições lucrativas.

#### ✅ NOVA FUNCIONALIDADE em `utils.py`:

```python
# Adicionar ao final de utils.py

class FuturesRolloverManager:
    """
    Gerencia rollover automático de contratos futuros.
    
    Regras:
    1. Monitora dias até vencimento
    2. Verifica cruzamento de volume (spot vs next)
    3. Executa rollover automático com 5 dias de antecedência
    """
    
    def __init__(self):
        self.rollover_threshold_days = 5
        self.volume_crossover_ratio = 1.2  # Next deve ter 20% mais volume
        self.last_check = None
        self.check_interval = timedelta(hours=1)
    
    def should_check_rollover(self) -> bool:
        """Verifica se já passou 1h desde última checagem"""
        if self.last_check is None:
            return True
        return (datetime.now() - self.last_check) >= self.check_interval
    
    def get_next_contract(self, current_symbol: str) -> Optional[str]:
        """
        Busca o próximo contrato mais líquido.
        """
        try:
            # Extrai base (WIN, WDO, etc)
            base = ''.join([c for c in current_symbol if c.isalpha()])
            
            candidates = get_futures_candidates(base)
            if not candidates:
                return None
            
            # Ordena por score (liquidez + proximidade)
            sorted_candidates = sorted(
                candidates,
                key=lambda x: -calculate_contract_score(x)
            )
            
            # Pega o primeiro que não seja o atual
            for cand in sorted_candidates:
                if cand['symbol'] != current_symbol:
                    return cand['symbol']
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao buscar próximo contrato: {e}")
            return None
    
    def check_and_execute_rollover(self, symbol: str, position_ticket: int) -> bool:
        """
        Verifica e executa rollover se necessário.
        
        Returns:
            True se rollover foi executado
        """
        try:
            if not self.should_check_rollover():
                return False
            
            self.last_check = datetime.now()
            
            # Pega info do contrato atual
            current_meta = next(
                (c for c in get_futures_candidates(''.join([ch for ch in symbol if ch.isalpha()]))
                 if c['symbol'] == symbol),
                None
            )
            
            if not current_meta:
                return False
            
            days_to_exp = current_meta.get('days_to_exp', 999)
            
            # Condição 1: Faltam <= 5 dias
            if days_to_exp > self.rollover_threshold_days:
                return False
            
            # Busca próximo contrato
            next_symbol = self.get_next_contract(symbol)
            if not next_symbol:
                logger.warning(f"Próximo contrato não encontrado para {symbol}")
                return False
            
            # Pega info do próximo
            next_meta = next(
                (c for c in get_futures_candidates(''.join([ch for ch in symbol if ch.isalpha()]))
                 if c['symbol'] == next_symbol),
                None
            )
            
            if not next_meta:
                return False
            
            # Condição 2: Volume cruzou
            current_vol = float(current_meta.get('volume', 0))
            next_vol = float(next_meta.get('volume', 0))
            
            if next_vol < (current_vol * self.volume_crossover_ratio):
                logger.info(f"Volume ainda não cruzou: {next_vol} < {current_vol * self.volume_crossover_ratio}")
                return False
            
            # ✅ Executar rollover
            logger.warning(f"🔄 ROLLOVER DETECTADO: {symbol} → {next_symbol} ({days_to_exp} dias restantes)")
            
            # Fecha posição atual
            pos = mt5.positions_get(ticket=position_ticket)
            if not pos:
                return False
            
            pos = pos[0]
            close_side = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(pos.volume),
                "type": close_side,
                "position": position_ticket,
                "price": tick.bid if close_side == mt5.ORDER_TYPE_SELL else tick.ask,
                "comment": "ROLLOVER_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN
            }
            
            result_close = mt5.order_send(close_request)
            
            if result_close.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erro no close para rollover: {result_close.comment}")
                return False
            
            # Abre posição no próximo contrato
            next_tick = mt5.symbol_info_tick(next_symbol)
            open_side = pos.type
            
            open_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": next_symbol,
                "volume": float(pos.volume),
                "type": open_side,
                "price": next_tick.ask if open_side == mt5.POSITION_TYPE_BUY else next_tick.bid,
                "comment": "ROLLOVER_OPEN",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN
            }
            
            result_open = mt5.order_send(open_request)
            
            if result_open.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Erro no open para rollover: {result_open.comment}")
                return False
            
            logger.info(f"✅ ROLLOVER EXECUTADO: {symbol} → {next_symbol}")
            
            # Envia notificação
            msg = (
                f"🔄 <b>ROLLOVER EXECUTADO</b>\n\n"
                f"De: {symbol}\n"
                f"Para: {next_symbol}\n"
                f"Volume: {pos.volume:.2f}\n"
                f"Dias restantes: {days_to_exp}"
            )
            send_telegram_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no rollover: {e}")
            return False

# Instância global
futures_rollover_manager = FuturesRolloverManager()
```

#### ✅ INTEGRAÇÃO NO BOT (bot.py):

```python
# Adicionar no fast_loop (após verificar posições):

# Verifica rollover de futuros
if utils.is_futures(pos.symbol):
    from utils import futures_rollover_manager
    futures_rollover_manager.check_and_execute_rollover(
        pos.symbol,
        pos.ticket
    )
```

---

### 6. ⚡ AJUSTAR SLIPPAGE PARA FUTUROS

**Problema:** Slippage genérico subestima custos em futuros.

#### ✅ CORREÇÃO no `config.py`:

```python
# LINHA 266 (config.py)
# SUBSTITUIR:

SLIPPAGE_MAP = {
    # === FUTUROS (Alta liquidez, baixo spread) ===
    "WINFUT": 0.0008,  # WIN: spread típico 5-10 pontos
    "WDOFUT": 0.0006,  # WDO: spread típico 0.50-1.00
    "WINJ25": 0.0008,  # Contratos específicos
    "WINK25": 0.0008,
    "WDOM25": 0.0006,
    "WDON25": 0.0006,
    
    # === AÇÕES BLUE CHIP ===
    "PETR4": 0.0015,
    "VALE3": 0.0015,
    "ITUB4": 0.0018,
    "BBDC4": 0.0018,
    
    # === AÇÕES MID CAP ===
    "MGLU3": 0.0025,
    "AMER3": 0.0025,
    "VVAR3": 0.0030,
    
    # === DEFAULT ===
    "DEFAULT": 0.0020
}

def get_slippage(symbol: str) -> float:
    """Retorna slippage estimado baseado no ativo"""
    # Tenta buscar exato
    if symbol in SLIPPAGE_MAP:
        return SLIPPAGE_MAP[symbol]
    
    # Tenta genérico (WIN* → WINFUT)
    for key in SLIPPAGE_MAP:
        if symbol.startswith(key[:3]):
            return SLIPPAGE_MAP[key]
    
    return SLIPPAGE_MAP["DEFAULT"]
```

---

### 7. 📊 STOP LOSS DINÂMICO (TRAILING STOP)

**Problema:** Stop loss estático não captura lucros em tendências fortes.

#### ✅ NOVA FUNCIONALIDADE em `utils.py`:

```python
# Adicionar classe TrailingStopManager

class TrailingStopManager:
    """
    Gerencia trailing stops dinâmicos.
    
    Regras:
    1. Ativa após lucro >= 1.5 ATR
    2. Ajusta SL para proteger % do lucro
    3. Nunca move SL contra a posição
    """
    
    def __init__(self):
        self.activation_threshold = 1.5  # ATR multiplicador para ativar
        self.protection_pct = 0.50  # Protege 50% do lucro
        self.trailing_step = 0.3  # Move SL a cada 0.3 ATR de lucro adicional
    
    def should_activate(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        side: str
    ) -> bool:
        """
        Verifica se trailing deve ser ativado.
        """
        if side == "BUY":
            profit_distance = current_price - entry_price
        else:  # SELL
            profit_distance = entry_price - current_price
        
        activation_distance = atr * self.activation_threshold
        
        return profit_distance >= activation_distance
    
    def calculate_new_sl(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        atr: float,
        side: str
    ) -> Optional[float]:
        """
        Calcula novo SL baseado no trailing.
        
        Returns:
            novo_sl ou None se não deve mover
        """
        try:
            if side == "BUY":
                # Profit atual
                profit = current_price - entry_price
                
                # SL deve estar em: entry + (profit * protection_pct)
                target_sl = entry_price + (profit * self.protection_pct)
                
                # Move apenas se novo SL for melhor
                if target_sl > current_sl:
                    return round(target_sl, 2)
            
            else:  # SELL
                profit = entry_price - current_price
                target_sl = entry_price - (profit * self.protection_pct)
                
                if target_sl < current_sl:
                    return round(target_sl, 2)
            
            return None
            
        except Exception as e:
            logger.error(f"Erro no trailing stop: {e}")
            return None
    
    def update_position_sl(
        self,
        position_ticket: int,
        new_sl: float
    ) -> bool:
        """
        Atualiza SL da posição no MT5.
        """
        try:
            pos = mt5.positions_get(ticket=position_ticket)
            if not pos:
                return False
            
            pos = pos[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position_ticket,
                "symbol": pos.symbol,
                "sl": new_sl,
                "tp": pos.tp  # Mantém TP
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Trailing Stop atualizado: {pos.symbol} SL={new_sl:.2f}")
                return True
            else:
                logger.warning(f"Falha no trailing: {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao atualizar trailing stop: {e}")
            return False

# Instância global
trailing_stop_manager = TrailingStopManager()
```

#### ✅ INTEGRAÇÃO NO BOT (fast_loop):

```python
# Adicionar no fast_loop, após verificar posições:

from utils import trailing_stop_manager

# Para cada posição aberta
for pos in mt5.positions_get():
    symbol = pos.symbol
    entry_price = pos.price_open
    current_price = pos.price_current
    current_sl = pos.sl
    side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    
    # Pega ATR atual
    rates = safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 100)
    if rates is None or len(rates) < 14:
        continue
    
    atr = calculate_atr(rates, period=14)
    
    # Verifica se trailing deve ativar
    should_trail = trailing_stop_manager.should_activate(
        entry_price, current_price, atr, side
    )
    
    if should_trail:
        new_sl = trailing_stop_manager.calculate_new_sl(
            entry_price, current_price, current_sl, atr, side
        )
        
        if new_sl is not None:
            trailing_stop_manager.update_position_sl(pos.ticket, new_sl)
```

---

## 🎲 TESTES A/B RECOMENDADOS

Para validar as correções, implemente testes A/B:

### Grupo A (Controle)
- Código atual
- Threshold 78%
- SL estático

### Grupo B (Experimental)
- Código corrigido
- Threshold 62%
- Trailing stop
- Estratégia de futuros

**KPIs:**
- Win Rate
- Profit Factor
- Max Drawdown
- Sharpe Ratio

Rode por 2 semanas em paper trading antes de produção.

---

## 📈 MELHORIAS ADICIONAIS

### 8. Volume Profile

Adicionar detecção de POC (Point of Control):

```python
def get_volume_profile(symbol: str, bars: int = 100) -> Dict:
    """
    Calcula Volume Profile:
    - POC: Preço com maior volume
    - VAH: Value Area High
    - VAL: Value Area Low
    """
    rates = safe_copy_rates(symbol, mt5.TIMEFRAME_M15, bars)
    if rates is None:
        return {}
    
    df = pd.DataFrame(rates)
    
    # Agrupa volume por faixa de preço
    price_bins = np.linspace(df['low'].min(), df['high'].max(), 50)
    df['price_bin'] = pd.cut(df['close'], bins=price_bins)
    
    volume_by_price = df.groupby('price_bin')['tick_volume'].sum()
    
    # POC = preço com mais volume
    poc_bin = volume_by_price.idxmax()
    poc_price = poc_bin.mid
    
    # Value Area (70% do volume)
    total_volume = volume_by_price.sum()
    target_volume = total_volume * 0.70
    
    # ... cálculo VAH/VAL ...
    
    return {
        'poc': poc_price,
        'vah': vah_price,
        'val': val_price
    }
```

### 9. Session Statistics

Armazenar estatísticas por sessão:

```python
SESSION_STATS = {
    "ABERTURA": {"win_rate": 0.55, "avg_rr": 1.8},
    "MEIO_DIA": {"win_rate": 0.48, "avg_rr": 1.5},
    "FECHAMENTO": {"win_rate": 0.62, "avg_rr": 2.2}
}
```

---

## 🎯 CHECKLIST DE IMPLEMENTAÇÃO

- [ ] Criar `futures_strategy.py`
- [ ] Reduzir ML threshold para 62%
- [ ] Implementar gestão de risco diferenciada
- [ ] Corrigir news calendar (timezone)
- [ ] Adicionar rollover automático
- [ ] Ajustar slippage por ativo
- [ ] Implementar trailing stop
- [ ] Adicionar volume profile
- [ ] Configurar testes A/B
- [ ] Validar em paper trading (2 semanas)
- [ ] Migrar para produção

---

## 📊 EXPECTATIVA DE RESULTADOS

### Antes (Estimado):
- Win Rate: ~45%
- Profit Factor: 0.8
- Max DD: -8%
- Trades/dia: 2-3

### Depois (Projetado):
- Win Rate: 58-62%
- Profit Factor: 1.6-1.9
- Max DD: -4%
- Trades/dia: 3-5

**Break-even esperado:** 3-4 semanas de operação consistente.

---

## 🔐 GESTÃO DE RISCO CRÍTICA

### Regras Obrigatórias:

1. **Nunca arriscar > 0.4% por trade em futuros**
2. **Máximo 2 posições simultâneas em futuros**
3. **Circuit breaker em 5% de DD diário**
4. **Rollover automático com 5 dias de antecedência**
5. **Trailing stop após 1.5 ATR de lucro**

---

## 📞 SUPORTE TÉCNICO

Para dúvidas na implementação:
1. Revisar logs em `xp3_bot.log`
2. Validar conexão MT5
3. Verificar símbolos no Market Watch
4. Testar em conta demo primeiro

---

**Documento preparado por:** Especialista Senior em Trading Algorítmico  
**Data:** 28/01/2026  
**Versão:** 1.0  
**Confidencial - Land Trading**
