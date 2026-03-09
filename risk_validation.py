import numpy as np

class BayesianRiskManager:
    """
    Motor de Gerenciamento de Risco baseado em Inferência Bayesiana e Kelly Fracionário.
    Ajusta o tamanho da posição dinamicamente com base no Regime do HMM e no ATR.
    """
    def __init__(self, base_win_rate=0.55, base_payout=1.5, kelly_fraction=0.10, tick_value=0.20, capital_allocation=1.0):
        # Prior (Nossa expectativa base extraída do Monte Carlo / Kalman)
        self.base_win_rate = base_win_rate
        self.base_payout = base_payout
        
        # O quanto do Kelly ótimo vamos usar (0.10 = 1/10th Kelly) - Foco em Sobrevivência
        self.kelly_fraction = kelly_fraction 
        self.tick_value = tick_value # WIN = R$ 0,20 por ponto
        self.capital_allocation = capital_allocation # Fatia do capital total (Portfólio)

    def get_posterior_win_rate(self, hmm_regime):
        """
        Atualiza a probabilidade de acerto (Posterior) com base na evidência (Regime HMM).
        Suporta 2 ou 3 estados.
        """
        # Mapeamento dinâmico:
        # Regime 0: Geralmente Consolidação/Ruído (Choppy)
        # Regime 1: Tendência (Bull ou Bear se 3 estados)
        # Regime 2: Tendência (Bear se 3 estados)
        
        if hmm_regime == 0:
            # Choppy: Proibitivo. Reduz expectativa drasticamente.
            posterior_p = self.base_win_rate * 0.50 
        elif hmm_regime in [1, 2]:
            # Tendência (Bull ou Bear): Alta confiança.
            posterior_p = min(self.base_win_rate * 1.30, 0.90) 
        else:
            posterior_p = self.base_win_rate
            
        return posterior_p

    def calculate_position_size(self, total_capital, hmm_regime, atr_points, confidence=1.0):
        """
        Calcula a quantidade exata de contratos a operar usando Kelly ajustado pelo ATR e pela Confiança da IA.
        total_capital: Capital total da conta.
        confidence: Probabilidade do estado atual (0.0 a 1.0).
        """
        # 0. Capital efetivo para este ativo (Fatia)
        effective_capital = total_capital * self.capital_allocation
        
        # 1. Obter Probabilidade Atualizada (Bayes)
        p = self.get_posterior_win_rate(hmm_regime)
        q = 1.0 - p
        b = self.base_payout

        # 2. Fórmula de Kelly
        kelly_optimal = (p * b - q) / b
        
        if kelly_optimal <= 0:
            return 0, 0.0, f"Expectativa Negativa (WR:{p:.1%}). Confiança IA: {confidence:.2%}. Kelly zerado."

        # 3. Kelly Fracionário Dinâmico (Risco % do capital ajustado pela confiança)
        # Scale: risk_pct * confidence (Se a IA estiver 50% segura, arriscamos metade do planejado)
        risk_pct = kelly_optimal * self.kelly_fraction * confidence
        risk_brl = effective_capital * risk_pct

        # 4. Risco Financeiro do Trade pelo ATR
        stop_loss_points = atr_points * 2 
        risk_per_contract = stop_loss_points * self.tick_value 
        
        if risk_per_contract <= 0:
            return 0, 0.0, "Erro: ATR inválido."

        # 5. Dimensionamento Final
        contracts = np.floor(risk_brl / risk_per_contract)
        
        # Filtro de segurança adicional: se for Regime 0 (Choppy), força zero.
        if hmm_regime == 0:
            contracts = 0 
            
        return int(contracts), risk_pct, f"Kelly_Opt: {kelly_optimal:.2%} | Confiança IA: {confidence:.2%} | WR: {p:.1%}"

# ==========================================
# SIMULAÇÃO DE DECISÃO EM TEMPO REAL
# ==========================================
if __name__ == "__main__":
    capital_atual = 10000.0
    atr_atual = 150.0 # Exemplo: Volatilidade atual de 150 pontos
    
    # Instancia o motor com Kelly Fracionário de 1/10
    risk_motor = BayesianRiskManager(base_win_rate=0.55, base_payout=1.5, kelly_fraction=0.1)
    
    print(f"💰 Capital: R$ {capital_atual} | ATR: {atr_atual} pontos")
    print(f"Prior Win Rate: {risk_motor.base_win_rate:.0%} | Payout: {risk_motor.base_payout}\n")
    
    # Simulando o Robô detectando o Regime 0 (Ruído/Consolidação)
    lote_r0, risco_r0, log_r0 = risk_motor.calculate_position_size(capital_atual, hmm_regime=0, atr_points=atr_atual)
    print(f"📉 HMM Detectou Regime 0 (Consolidação):")
    print(f"   -> Ação: Operar {lote_r0} contratos. ({log_r0})\n")
    
    # Simulando o Robô detectando o Regime 1 (Tendência)
    lote_r1, risco_r1, log_r1 = risk_motor.calculate_position_size(capital_atual, hmm_regime=1, atr_points=atr_atual)
    print(f"📈 HMM Detectou Regime 1 (Volatilidade/Tendência):")
    print(f"   -> Ação: Operar {lote_r1} contratos. Risco Alocado: {risco_r1:.2%} do capital. ({log_r1})")
    
    # Teste com ATR dobrado (Alta volatilidade na tendência)
    atr_alto = 300.0
    lote_vol, risco_vol, log_vol = risk_motor.calculate_position_size(capital_atual, hmm_regime=1, atr_points=atr_alto)
    print(f"\n⚡ Alta Volatilidade detectada (ATR: {atr_alto} pontos):")
    print(f"   -> Ação: Operar {lote_vol} contratos. (Lote reduzido para equilibrar risco financeiro)")
