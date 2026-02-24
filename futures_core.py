# futures_core.py
# Núcleo de Lógica para Futuros B3 (Detecção, Rollover, Dados)

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
import config_futures

logger = logging.getLogger("futures_core")

class FuturesContract:
    def __init__(self, base_symbol, expiry_code, expiry_date):
        self.base = base_symbol          # 'WIN'
        self.code = expiry_code           # 'G26'
        self.full_symbol = f"{base_symbol}{expiry_code}"  # 'WING26'
        self.expiry = expiry_date
        
    def days_to_expiry(self):
        if not isinstance(self.expiry, datetime):
            return 999
        return (self.expiry - datetime.now()).days
    
    def is_near_expiry(self, threshold=5):
        return self.days_to_expiry() <= threshold

class FuturesDataManager:
    def __init__(self, mt5_connection=None):
        self.mt5 = mt5_connection if mt5_connection else mt5
        self.contracts_cache = {}
    
    def _parse_expiry(self, symbol):
        """Tenta extrair data de expiração do MT5"""
        try:
            info = self.mt5.symbol_info(symbol)
            if info and info.expiration_time:
                return datetime.fromtimestamp(info.expiration_time)
        except:
            pass
        return datetime.now() + timedelta(days=365) # Fallback

    def find_front_month(self, base_symbol):
        """
        Detecta contrato mais líquido automaticamente
        SOLUÇÃO 2: Filtra por Open Interest e Validação de Vencimento
        """
        # 1. Lista todos contratos disponíveis usando Regex
        pattern_str = f"{base_symbol}[FGHJKMNQUVXZ]\\d{{2}}"
        all_symbols = self.mt5.symbols_get(group=f"*{base_symbol}*")
        
        if not all_symbols:
            logger.warning(f"Nenhum simbolo encontrado para base {base_symbol}")
            return None

        candidates = []
        regex = re.compile(pattern_str)
        # Fix: config keys use "WIN$N" format, not "WIN"
        config_key = f"{base_symbol}$N"
        min_oi = config_futures.FUTURES_CONFIGS.get(config_key, {}).get('min_oi', 1000)
        now = datetime.now()

        for s in all_symbols:
            if not regex.search(s.name):
                continue
                
            info = self.mt5.symbol_info(s.name)
            if not info:
                continue
            
            # SOLUÇÃO 10: Liquidez por OI
            oi = getattr(info, "session_open_interest", None)
            if oi in (None, 0):
                oi = getattr(info, "open_interest", None)
            
            # Fallback 1: Volume
            if oi in (None, 0):
                volume = max(
                    float(getattr(info, "volume", 0) or 0),
                    float(getattr(info, "volumehigh", 0) or 0)
                )
                # Se tem volume, usa como proxy de liquidez
                if volume > 0:
                    oi = volume
            
            # Assegura que OI é float para ordenação
            oi = float(oi or 0)

            try:
                exp_timestamp = info.expiration_time
                if exp_timestamp and exp_timestamp > 86400: # Maior que 1 dia desde epoch
                    exp_date = datetime.fromtimestamp(exp_timestamp)
                else:
                    exp_date = now + timedelta(days=365) # Fallback seguro
            except Exception:
                exp_date = now + timedelta(days=365) # Fallback seguro em caso de erro
            
            # Filtra expirados (margem de segurança de 1 dia)
            # Deve ser FUTURO: exp_date >= now
            days_until_exp = (exp_date - now).days
            if days_until_exp < 0:
                continue

            candidates.append({
                'symbol': s.name,
                'oi': oi,
                'expiry': exp_date,
                'days_to_expiry': days_until_exp
            })
        
        if not candidates:
            return None
            
        # SOLUÇÃO 2, ETAPA 2: Filtrar
        # Se todos tiverem OI=0, ordena APENAS por data de expiração (menor 'days_to_expiry' positivo)
        total_oi = sum(c['oi'] for c in candidates)
        
        if total_oi == 0:
            # Fallback total: Pega o contrato com vencimento mais próximo (mas não hoje/ontem)
            # Idealmente entre 5 e 45 dias para evitar contratos prestes a vencer ou muito longos
            candidates.sort(key=lambda x: x['days_to_expiry'])
            best = candidates[0]
            logger.warning(f"⚠️ Sem dados de OI/Vol para {base_symbol}. Usando {best['symbol']} pelo vencimento ({best['days_to_expiry']} dias).")
            return best['symbol']

        # Ordena por OI (desc), depois por Expiry (asc)
        candidates.sort(key=lambda x: (-x['oi'], x['days_to_expiry']))
        
        # Pega o top e verifica se é valido
        best = candidates[0]
        
        # Se OI muito baixo, talvez warning?
        if best['oi'] < min_oi:
            logger.warning(f"⚠️ Contrato {best['symbol']} selecionado com liquidez baixa (OI/Vol={best['oi']}) para {base_symbol}.")
        
        # SOLUÇÃO 2, ETAPA 3: Check de Rollover
        if (best['expiry'] - now).days <= 4:
            if len(candidates) > 1:
                second = candidates[1]
                # Se o segundo já tem > 50% do OI do primeiro, pode ser hora de mudar
                if second['oi'] > best['oi'] * 0.5:
                     logger.info(f"🔄 Rollover iminente: {best['symbol']} -> {second['symbol']}")
                     return second['symbol']
        
        return best['symbol']
    
    def get_contract_chain(self, base_symbol, months=24):
        """
        Retorna lista de contratos passados e futuros ordenados por data
        Útil para concatenar histórico.
        """
        # Implementação simplificada: buscar no MT5 todos que match regex
        all_symbols = self.mt5.symbols_get(group=f"*{base_symbol}*")
        if not all_symbols: return []
        
        pattern_str = f"{base_symbol}[FGHJKMNQUVXZ]\\d{{2}}"
        regex = re.compile(pattern_str)
        
        chain = []
        for s in all_symbols:
            if regex.search(s.name):
                info = self.mt5.symbol_info(s.name)
                exp = datetime.fromtimestamp(info.expiration_time) if info.expiration_time else datetime.now()
                chain.append({'symbol': s.name, 'expiry': exp})
        
        # Ordena por expiração
        chain.sort(key=lambda x: x['expiry'])
        return [c['symbol'] for c in chain]

    def concatenate_history(self, base_symbol, bars=5000, timeframe=mt5.TIMEFRAME_M15):
        """
        SOLUÇÃO 9: Tratamento de Rollover (Gap Correction)
        Constrói série histórica contínua com rollover
        """
        contracts = self.get_contract_chain(base_symbol)
        
        front_month = self.find_front_month(base_symbol)
        if not front_month: return None
        
        all_data = []
        
        # Itera de trás para frente para ajustar os gaps
        for i in range(len(contracts) - 1, 0, -1):
            current_contract = contracts[i]
            prev_contract = contracts[i-1]
            
            # Pega dados do contrato atual
            current_df = self._get_data(current_contract, bars, timeframe)
            if current_df is None or current_df.empty:
                continue

            # Pega último dia do contrato anterior para calcular o gap
            prev_df_last_day = self._get_data(prev_contract, 10, timeframe) # Pega algumas barras pra garantir
            if prev_df_last_day is None or prev_df_last_day.empty:
                all_data.insert(0, current_df)
                continue

            # Achar ponto de junção (último tick do anterior)
            junction_time = prev_df_last_day.index[-1]
            
            # Filtra o df atual para pegar o primeiro tick após a junção
            future_bars = current_df.loc[current_df.index > junction_time]
            if future_bars.empty:
                # Não há dados novos após a junção, pula o ajuste de gap
                all_data.insert(0, current_df)
                continue

            # Gap de fechamento
            gap = future_bars['open'].iloc[0] - prev_df_last_day['close'].iloc[-1]
            
            # Ajusta o histórico anterior
            if i-1 < len(all_data):
                all_data[i-1][['open', 'high', 'low', 'close']] -= gap

            all_data.insert(0, current_df)

        if not all_data:
            return self._get_data(front_month, bars, timeframe)

        full_history = pd.concat(all_data)
        full_history.drop_duplicates(inplace=True)
        
        return full_history.tail(bars) 

    def _get_data(self, symbol, n_bars, timeframe):
        if not self.mt5.symbol_select(symbol, True):
            logger.warning(f"Falha ao selecionar {symbol} no MT5")
        
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None: return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # ✅ FIX: Renomeia volume corretamente
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        elif 'real_volume' in df.columns:
            df.rename(columns={'real_volume': 'volume'}, inplace=True)
            
        df.set_index('time', inplace=True)
        return df

    def volume_decay_factor(self, df, symbol):
        """
        SOLUÇÃO 12: Volume Ponderado por Tempo até Vencimento
        """
        info = self.mt5.symbol_info(symbol)
        if not info or not info.expiration_time: return df['volume']
        
        exp_date = datetime.fromtimestamp(info.expiration_time)
        
        # Calcula dias até vencimento para cada barra
        days_to_exp = (exp_date - df.index).days
        
        # Clip days em max 30 para não aumentar volume absurdamente antes
        days_clipped = np.minimum(days_to_exp, 30)
        # E clip min para não dar erro
        days_clipped = np.maximum(days_clipped, 0)
        
        arg = -0.15 * (30 - days_clipped)
        decay = np.exp(arg)
        
        return df['volume'] * decay

# Helper global
def get_manager():
    return FuturesDataManager(mt5)
