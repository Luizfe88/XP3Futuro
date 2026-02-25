# Otimizador Sklearn: Documentação Técnica

## Visão Geral
Sistema de otimização para modelos scikit-learn com:
- Validação cruzada consistente (StratifiedKFold, seed fixa)
- Seleção automática de hiperparâmetros (GridSearchCV, RandomizedSearchCV)
- Três métodos de otimização avançados: Gradient Descent, Genetic Algorithms, Simulated Annealing
- Métricas: melhor score CV, tempo de execução, histórico e curva de convergência
- Logging detalhado, tratamento robusto de exceções, reprodutibilidade

## Arquitetura
- `ml_optimizer.py` concentra:
  - Wrappers de Grid/Random Search
  - Implementações de GD/GA/SA para espaços paramétricos numéricos e categóricos
  - `OptimizationResult` para padronizar saída (params/score/tempo/convergência/history)
  - `build_default_pipeline`: Pipeline(StandardScaler, PCA opcional, Estimator)
  - `compare_methods`: executa todos e retorna dict de resultados

## Métodos de Otimização
### Gradient Descent (GD)
- Aproxima gradiente por diferenças finitas sobre o score de CV
- Atualização: `param = clamp(param + step * grad)` com `step = lr / sqrt(iter)`
- Espaço suportado: contínuo (min/max, opcional log) e categórico (mantido)
- Métricas: histórico por iteração, best_score/params, tempo total

### Genetic Algorithms (GA)
- População inicial amostrada do espaço
- Seleção por fitness (score CV), elitismo, crossover e mutação
- Mutação: ruído normal em contínuos, re-amostragem em categóricos
- Métricas: best por geração, convergência, tempo total

### Simulated Annealing (SA)
- Vizinhança univariada aleatória por iteração
- Critério de aceitação por probabilidade `exp((Δscore)/T)` com resfriamento exponencial
- Suporta espaços contínuos/categóricos
- Métricas: histórico corrente, melhor score, curva de convergência

## Seleção de Hiperparâmetros Sklearn
- `GridSearchCV`: grade discreta de parâmetros
- `RandomizedSearchCV`: distribuições/linspace; `random_state` fixo
- Ambos usam `StratifiedKFold` e `scoring` configurável (padrão accuracy)

## Pipelines e Transformações
- `StandardScaler` por padrão; `PCA` opcional (parametrizável via pipeline)
- Estimadores compatíveis: qualquer classe sklearn com `fit/predict`
- Parametrização segue convenção `model__param` e `pca__param`

## Reprodutibilidade
- Seeds fixas em: gerador numpy, StratifiedKFold, estimadores e PCA
- Resultados de GA/SA e RandomizedSearch ficam determinísticos dado `random_state`

## Logging e Tratamento de Exceções
- INFO: início/fim e progresso (iterações/gerações/parciais)
- DEBUG: falhas pontuais de avaliação sem interromper execução
- Exceções capturadas e registradas; métodos continuam quando possível

## Métricas de Performance
- `best_score`: média de `cross_val_score`
- `runtime_seconds`: tempo total do método
- `convergence`: lista de (iter/geração, melhor score)
- `history`: detalhes por iteração/geração (parâmetros e score)

## Interface Visual
- `dashboard.py`: comparador de métodos com dataset sintético
- Exibe tabela de Score/Tempo e gráfico de barras; mostra melhores parâmetros

## Testes Unitários
- `tests/test_ml_optimizer.py` cobre:
  - Grid/Random Search (scores válidos e params)
  - Reprodutibilidade de GA
  - Smoke test `compare_methods`

## Exemplos de Uso
Consulte `compare_methods` para execução rápida com `LogisticRegression` e espaços `grid/dist/numeric`.

## Extensões Futuras
- Suporte a métricas adicionais (AUC, F1), pipelines customizados (ColumnTransformer)
- Espaços condicionais e bayesian optimization (Optuna/Skopt) se necessário
