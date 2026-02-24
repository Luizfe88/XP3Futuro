# XP3 Future Bot

Sistema de trading automatizado para futuros da B3 com integração nativa ao MetaTrader 5, filtros de risco, gestão dinâmica, mapeamento automático de contratos e suporte opcional a sinais de ML (ensemble).

## Visão Geral

- Opera contratos futuros (WIN, WDO, BGI, WSP, etc.) com regras de risco e janelas de mercado.
- Integra MetaTrader 5 para dados e execução.
- Mapeia e habilita automaticamente contratos ativos, com fallback inteligente por corretora.
- Aplica filtros: liquidez, ATR, spread, notícias (blackout), correlação e hedging.
- Configuração dinâmica via YAML e ambiente (.env), com suporte a ajustes em runtime.

## Principais Funcionalidades

- Mapeamento automático de futuros e fallback por corretora (XP: sufixos N/Z; outras: formato tradicional) [utils.py](file:///c:/Users/luizf/Documents/xp3future/utils.py).
- Descoberta e ativação de ativos do dia com filtros de qualidade [utils.py](file:///c:/Users/luizf/Documents/xp3future/utils.py#L537-L605).
- Filtro de notícias e blackout de eventos [news_calendar.py](file:///c:/Users/luizf/Documents/xp3future/news_calendar.py).
- Cache de dados com Redis para performance [utils.py](file:///c:/Users/luizf/Documents/xp3future/utils.py#L1193-L1201).
- Gestão de risco e horários de operação [config.py](file:///c:/Users/luizf/Documents/xp3future/config.py#L223-L241).
- Painel/relatórios auxiliares via dashboard.py e logs estruturados.

## Arquitetura

- Bot principal: [bot.py](file:///c:/Users/luizf/Documents/xp3future/bot.py) — orquestra execução, estados e rotinas diárias.
- Utilidades e integrações MT5/Redis: [utils.py](file:///c:/Users/luizf/Documents/xp3future/utils.py).
- Configurações dinâmicas: [config.py](file:///c:/Users/luizf/Documents/xp3future/config.py) + [config.yaml](file:///c:/Users/luizf/Documents/xp3future/config.yaml).
- Lógica de futuros: [futures_core.py](file:///c:/Users/luizf/Documents/xp3future/futures_core.py).
- Aquisição de dados e métricas: [market_screener.py](file:///c:/Users/luizf/Documents/xp3future/market_screener.py), [metrics.py](file:///c:/Users/luizf/Documents/xp3future/metrics.py).
- Filtro de notícias: [news_calendar.py](file:///c:/Users/luizf/Documents/xp3future/news_calendar.py), [news_filter.py](file:///c:/Users/luizf/Documents/xp3future/news_filter.py).
- Hedging e correlações: [hedging.py](file:///c:/Users/luizf/Documents/xp3future/hedging.py).
- Sinais de ML (opcional): [ml_signals.py](file:///c:/Users/luizf/Documents/xp3future/ml_signals.py), [ml_optimizer.py](file:///c:/Users/luizf/Documents/xp3future/ml_optimizer.py).

## Requisitos

- Python 3.10+ (recomendado).
- MetaTrader 5 instalado (Terminal 64 bits) e credenciais de conta.
- Redis em execução local (padrão: localhost:6379) para cache.
- Pacotes Python (instale via pip):
  - MetaTrader5, pandas, numpy, requests, joblib, pytz
  - redis (para cache)
  - Opcional/ML: scikit-learn, xgboost, tensorflow/keras (para modelos em [models/](file:///c:/Users/luizf/Documents/xp3future/models))

## Instalação

1. Crie um ambiente virtual:
   - Windows (PowerShell): `python -m venv .venv` e `.\.venv\Scripts\Activate.ps1`
2. Instale dependências:
   - `pip install MetaTrader5 pandas numpy requests joblib pytz redis`
   - Se usar ML: `pip install scikit-learn xgboost tensorflow`
3. Ajuste o caminho do terminal MT5 em [config.yaml](file:///c:/Users/luizf/Documents/xp3future/config.yaml) (`mt5.terminal_path`).

## Configuração

- Arquivo YAML: [config.yaml](file:///c:/Users/luizf/Documents/xp3future/config.yaml)
  - `mt5.terminal_path`: caminho do terminal 64 bits.
  - `risk_levels`: parâmetros por perfil (CONSERVADOR, MODERADO, AGRESSIVO).
  - `ml` e `trading`: enable/disable de funcionalidades.
- Variáveis de ambiente: [.env](file:///c:/Users/luizf/Documents/xp3future/.env)
  - Defina `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MODE` (DEMO/REAL).
  - Nunca versionar credenciais reais.
- Config dinâmico em runtime via [config.py](file:///c:/Users/luizf/Documents/xp3future/config.py):
  - `ConfigManager` carrega/salva YAML e expõe valores legados (ex.: `MT5_TERMINAL_PATH`).

## Execução

- Bot principal:
  - `python bot.py`
- Utilitários:
  - Diagnóstico de feed MT5: `python check_mt5_feed.py`
  - Dashboard/monitoramento: `python dashboard.py`
  - Debug de futuros: `python debug_futures_loading.py`

## Mapeamento de Futuros

- Descoberta geral com fallback:
  - [discover_all_futures](file:///c:/Users/luizf/Documents/xp3future/utils.py#L475-L507): tenta mapear `WIN$`, `WDO$`, `WSP$`, `BGI$`, etc.; aplica fallback se nenhum candidato for encontrado.
- Fallback por corretora:
  - [\_fallback_future_symbol](file:///c:/Users/luizf/Documents/xp3future/utils.py#L855-L867): 
    - XP: usa sufixos `N`/`Z` (ex.: `WINZ`, `WDOZ`) conforme paridade do mês.
    - Outras: usa formato tradicional (ex.: `WING26`, `WDOG26`).
- Ativação com filtros:
  - [find_and_enable_active_futures](file:///c:/Users/luizf/Documents/xp3future/utils.py#L508-L605): escolhe contrato ativo por volume e aplica filtros (liquidez, ATR, spread), habilitando no Market Watch.

## Logs e Artefatos

- Logs de bot: [logs/bot/](file:///c:/Users/luizf/Documents/xp3future/logs/bot).
- Logs de erros: [logs/errors/](file:///c:/Users/luizf/Documents/xp3future/logs/errors).
- Relatórios de análise: [logs/analysis/](file:///c:/Users/luizf/Documents/xp3future/logs/analysis).
- Mapeamentos de futuros: [futures_optimizer_output/](file:///c:/Users/luizf/Documents/xp3future/futures_optimizer_output).
- Estados/artefatos diários: arquivos como `daily_bot_state.json`, `daily_symbol_limits.json`.

## Troubleshooting

- MT5 não conecta:
  - Verifique `mt5.terminal_path` no YAML e credenciais no `.env`.
  - Execute `python check_mt5_feed.py` para diagnosticar.
  - Garanta que o terminal esteja logado e com permissão de negociação.
- Sem candidatos de futuros:
  - Use `discover_all_futures` para fallback e confirme corretora detectada nos logs.
  - Confirme que símbolos estão visíveis no Market Watch.
- Redis indisponível:
  - Inicie `redis-server` local ou ajuste host/porta em [utils.py](file:///c:/Users/luizf/Documents/xp3future/utils.py#L1193-L1201).

## Pastas e Arquivos Importantes

- Código fonte principal na raiz do projeto.
- Modelos ML: [models/](file:///c:/Users/luizf/Documents/xp3future/models) (ex.: `lstm_signal.h5`, `xgb_signal.pkl`).
- Métricas: [metrics/](file:///c:/Users/luizf/Documents/xp3future/metrics).
- Cache/recursos: [cache/](file:///c:/Users/luizf/Documents/xp3future/cache), [lsdata/](file:///c:/Users/luizf/Documents/xp3future/lsdata).

## Segurança

- Nunca publique credenciais reais no repositório.
- `.env` deve conter apenas placeholders; mantenha valores sensíveis fora do controle de versão.
- Evite logar senhas/segredos; revise handlers de logging conforme necessário.

## Referências de Código

- Descoberta de futuros com fallback: [discover_all_futures](file:///c:/Users/luizf/Documents/xp3future/utils.py#L475-L507)
- Fallback por corretora (XP N/Z, tradicional): [\_fallback_future_symbol](file:///c:/Users/luizf/Documents/xp3future/utils.py#L855-L867)
- Filtros e habilitação de ativos: [find_and_enable_active_futures](file:///c:/Users/luizf/Documents/xp3future/utils.py#L508-L605)
- Configuração dinâmica: [ConfigManager](file:///c:/Users/luizf/Documents/xp3future/config.py#L29-L99)

# XP3Futuro
