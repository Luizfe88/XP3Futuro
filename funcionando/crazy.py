# crazy_auto_final_v5.py – Último teste com precisão total do preço
import MetaTrader5 as mt5
import time
import sys

# --- ATENÇÃO: REMOVEMOS O LOGIN EXPLÍCITO ---
# O script agora confia que o MT5 está aberto e logado na conta correta.
SYMBOL_TO_TRADE = "PETR4" 
BASE_LOT_SIZE = 100    

# --- Inicialização ---
def initialize_mt5():
    if not mt5.initialize():
        print("ERRO: MT5 init failed", mt5.last_error())
        sys.exit()

    acc = mt5.account_info()
    if acc is None:
        print("ERRO: NENHUMA CONTA CONECTADA – Abra o MT5 e faça login na conta REAL/DEMO")
        mt5.shutdown()
        sys.exit()

    print(f"Login OK: {acc.name} (Login: {acc.login}) | Server: {acc.server}")
    if not acc.trade_allowed:
        print("ERRO: NEGOCIAÇÃO BLOQUEADA – Ative em Ferramentas → Opções → Expert Advisors no MT5")
        mt5.shutdown()
        sys.exit()
    
def execute_trade_no_sltp(symbol, order_type, lot_size_lotes):
    if not mt5.symbol_select(symbol, True):
        print(f"ERRO: Símbolo {symbol} não encontrado.")
        return None

    # Busca informações do símbolo para garantir volume correto
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"ERRO: Não foi possível obter info de {symbol}")
        return None

    # Garante que o volume respeite o incremento mínimo (Ex: 100 em 100)
    # Passo 1: O volume deve ser float
    volume = float(lot_size_lotes)
    
    # Passo 2: Ajusta para o passo permitido pelo servidor (volume_step)
    step = info.volume_step
    volume = round(volume / step) * step

    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,             # Agora garantido como float e múltiplo do step
        "type": order_type,
        "price": price,
        "deviation": 100,
        "magic": 777777,
        "comment": "CORRECAO_VOLUME",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    print(f"Enviando Ordem: {symbol} | Qtd: {volume} | Preço: {price}")
    result = mt5.order_send(request)
    
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = mt5.last_error() if result is None else result.comment
        print(f"FALHA: {err_msg}")
        return None

    return result


def run_auto_trade():
    """
    Executa a compra automática sem SL/TP e fecha em seguida (para segurança).
    """
    initialize_mt5()
    print("Iniciando compra automática...")
    result = execute_trade_no_sltp(SYMBOL_TO_TRADE, mt5.ORDER_TYPE_BUY, BASE_LOT_SIZE)

    if result:
        print(f"\nCOMPRA BEM-SUCEDIDA! Uma posição de {SYMBOL_TO_TRADE} foi aberta sem SL/TP.")
        position_id = result.order
        time.sleep(2)
        
        # Fecha a posição
        close_price = mt5.symbol_info_tick(SYMBOL_TO_TRADE).bid # Preço de fechamento com precisão total
        r_close = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL_TO_TRADE,
            "volume": BASE_LOT_SIZE,
            "type": mt5.ORDER_TYPE_SELL,
            "price": close_price,
            "position": position_id,
            "deviation": 50, 
            "magic": 777777,
            "type_filling": mt5.ORDER_FILLING_IOC, 
            "type_time": mt5.ORDER_TIME_GTC,
        })
        
        if r_close and r_close.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"FECHAMENTO OK! Posição {position_id} zerada.")
        else:
            print("AVISO: Falha ao fechar a posição. Feche-a manualmente no MT5.")
    else:
        print("\nFalha crítica na compra.")

    time.sleep(5)
    
if __name__ == "__main__":
    run_auto_trade()
    mt5.shutdown()