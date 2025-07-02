# VictimSim2 – Sistema Multiagente para Resgate de Vítimas

## Descrição Geral
Este projeto implementa um Sistema Multiagente (SMA) para simulação de resgate de vítimas em cenários de catástrofes naturais, desastres ou grandes acidentes, utilizando o simulador VictimSim2. O sistema é composto por dois grupos de agentes artificiais (robôs terrestres): exploradores e socorristas, que atuam de forma sequencial para localizar e socorrer vítimas em um ambiente dinâmico e parcialmente desconhecido.

---

## 1. Arquitetura do Sistema

- **Exploradores**: Localizam vítimas e constroem mapas parciais do ambiente, considerando restrições de bateria e obstáculos.
- **Sincronização**: Unificação dos mapas individuais em um único mapa global, compartilhado com os socorristas.
- **Socorristas**: Recebem o mapa unificado, agrupam as vítimas, definem sequências de resgate otimizadas e realizam o socorro, retornando à base antes do fim da bateria.

O sistema integra algoritmos de busca, clustering, otimização (AG ou busca local), classificação e regressão, além de técnicas de explicabilidade (LIME/SHAP).

---

## 2. Execução do Sistema

### Cenário de Referência
- **Grid**: 90x90
- **Vítimas**: 400
- **Configuração dos agentes**: `explorer_config.txt` e `rescuer_config.txt` (em `datasets/data_400v_90x90/`)
- **Dificuldade de acesso**: Intervalo (0, 3], obstáculos intransponíveis = 100 (`VS.OBST_WALL`)
- **Ambiente dinâmico**: Dificuldades podem variar durante a simulação.

### Como executar
1. **Configuração**: Ajuste os arquivos de configuração dos agentes conforme desejado.
2. **Execução**: Utilize os scripts principais em `mas/` ou `run_all_datasets.py` para rodar simulações.
3. **Resultados**: Saídas são geradas em diretórios específicos, como `Results_300v_90x90_cfg_1/` ou `Results_408v_94x94/`.

---

## 3. Componentes e Algoritmos

### 3.1 Exploração

O explorador implementa um **algoritmo híbrido inteligente** que combina múltiplas técnicas para maximizar a eficiência na descoberta de vítimas:

#### Algoritmos Utilizados:

**1. Busca em Largura (BFS - Breadth-First Search)**
- **Exploração sistemática**: Garante que todas as células acessíveis sejam visitadas
- **Cálculo de caminhos**: Encontra o caminho mais curto para posições específicas
- **Navegação para a base**: Calcula o caminho de volta otimizado
- **Implementação**: Utiliza `deque` para fronteira e `set` para posições visitadas

**2. Algoritmo de Mapa de Calor (Heat Map)**
- **Priorização inteligente**: Direciona a exploração para áreas com maior probabilidade de vítimas
- **Atualização dinâmica**: Células com vítimas recebem pontuação +2.0, células vizinhas +0.5
- **Penalização gradual**: Células sem vítimas perdem 0.1 pontos, vizinhas perdem 0.05
- **Adaptação contínua**: O mapa se ajusta conforme novas vítimas são descobertas

**3. Algoritmo de Detecção de Clusters**
- **Identificação de padrões**: Agrupa vítimas próximas para focar a exploração
- **Busca conectada**: Utiliza BFS para encontrar vítimas adjacentes
- **Estratégia de concentração**: Prioriza exploração em áreas com múltiplas vítimas

**4. Algoritmo de Priorização Inteligente**
- **Ordenação da fronteira**: Posições são ordenadas pelo valor do heat map
- **Cálculo de score**: Combina heat map com distância (score = heat - distância × 0.1)
- **Seleção adaptativa**: Escolhe a próxima posição baseada em múltiplos critérios

#### Características do Algoritmo Híbrido:

- **Exploração Sistemática**: BFS garante cobertura completa do ambiente
- **Priorização Inteligente**: Heat map direciona para áreas promissoras
- **Detecção de Padrões**: Clusters identificam concentrações de vítimas
- **Navegação Eficiente**: Cálculo de caminhos otimizados para qualquer posição
- **Adaptação Dinâmica**: O algoritmo se ajusta conforme encontra vítimas
- **Retorno Garantido**: Cada explorador retorna à base antes do fim da bateria
- **Unificação de Mapas**: Mapas individuais são consolidados em um mapa global

### 3.2 Agrupamento de Vítimas (Clustering)

#### Algoritmo K-Means Implementado
- **Função**: `cluster_victims()` no agente Rescuer
- **Propósito**: Agrupa as vítimas em clusters baseado em suas posições geográficas
- **Características**:
  - Máximo de 4 clusters (limitado pelo número de socorristas)
  - Usa distância euclidiana para calcular proximidade entre vítimas
  - Inicialização aleatória dos centróides
  - Máximo de 4 iterações para convergência
  - Adaptação dinâmica: número de clusters baseado no número de vítimas encontradas
- **Arquivos gerados**: `cluster1.txt`, `cluster2.txt`, ... (em `mas/clusters/` ou diretórios de resultados)
- **Formato**: `id, x, y, grav, classe`

### 3.3 Sequenciamento de Resgate

#### Algoritmo Genético Implementado
- **Função**: `genetic_algorithm()` no agente Rescuer
- **Propósito**: Otimiza a sequência de visita das vítimas (problema do caixeiro viajante)
- **Parâmetros**:
  - População: 100 indivíduos
  - Gerações: 50 iterações
  - Probabilidade de crossover: 80%
  - Probabilidade de mutação: 20%
- **Características**:
  - Seleção por torneio (3 candidatos por seleção)
  - Crossover com gene strip (preserva segmentos dos pais)
  - Mutação por troca de genes
  - Função de fitness baseada na distância total da sequência
  - Restrições de bateria e tempo consideradas
- **Arquivos gerados**: `seq1.txt`, `seq2.txt`, ... (em `mas/clusters/` ou diretórios de resultados)
- **Formato**: `id, x, y, grav, classe`

### 3.4 Execução do Socorro

#### Algoritmos de Busca de Caminho Implementados

**1. Algoritmo A*** 
- **Função**: `a_star()` no agente Rescuer
- **Propósito**: Encontra o caminho mais curto entre duas coordenadas no mapa
- **Características**:
  - Heurística de distância diagonal otimizada
  - Considera dificuldade de acesso das células do mapa
  - Custo diferenciado: movimentos diagonais (1.5) vs lineares (1.0)
  - Fila de prioridade para otimização da busca
  - Ajuste pela dificuldade mínima do mapa

**2. Breadth-First Search (BFS)**
- **Função**: `planner()` (usa classe BFS importada)
- **Propósito**: Calcula caminhos entre vítimas para planejamento offline
- **Características**:
  - Planejamento offline completo antes da execução
  - Considera custos lineares e diagonais configuráveis
  - Garante retorno à base após resgate de todas as vítimas
  - Otimização de tempo de execução

**3. Heurística de Distância Diagonal**
- **Função**: `heuristic()` no agente Rescuer
- **Propósito**: Estima custo entre duas coordenadas para o algoritmo A*
- **Características**:
  - Considera movimentos diagonais mais eficientes
  - Ajustada pela dificuldade mínima do mapa
  - Otimização para ambientes com obstáculos

- Cada socorrista segue a sequência definida, socorrendo o máximo de vítimas possível e retornando à base antes do fim da bateria.

### 3.5 Classificação e Regressão

#### Machine Learning para Classificação de Severidade
- **Função**: `predict_severity_and_class()` no agente Rescuer
- **Propósito**: Classifica a severidade das vítimas usando modelo treinado
- **Características**:
  - Carrega classificador pré-treinado (pickle)
  - Usa scaler para normalização dos dados
  - Prediz classe de severidade (1=crítico, 2=instável, 3=potencialmente estável, 4=estável)
  - Fallback para valores aleatórios se modelo não carregar
  - Baseado em sinais vitais: pSist, pDiast, qPA, pulso, resp

#### Modelos Implementados
- **Classificador**: Estima a classe de gravidade da vítima (1=crítico, 2=instável, 3=potencialmente estável, 4=estável).
- **Regressor**: Estima o valor contínuo de gravidade [0, 100].
- **Modelos testados**: CART e Redes Neurais (MLP), com 3 configurações cada.
- **Pré-processamento**: Idêntico para ambos os métodos, sem uso de variáveis proibidas.
- **Validação cruzada**: Utilizada para avaliação robusta dos modelos.
- **Testes cegos**: Realizados com o dataset de 800 vítimas.
- **Arquivos de modelos**: Em `mas/models/` (ex: `cart_classifier.pkl`, `mlp_regressor.pkl`)
- **Relatórios e comparações**: Em `mas/models/` (ex: `algorithm_comparison_report.txt`, `prediction_report.txt`)

### 3.6 Fluxo de Execução dos Algoritmos

#### Sequência de Processamento
1. **Exploração**: Exploradores mapeiam o ambiente usando BFS e heat map
2. **Clustering**: K-Means agrupa vítimas por proximidade geográfica
3. **Classificação**: ML classifica severidade das vítimas usando modelo treinado
4. **Sequenciamento**: Algoritmo genético otimiza ordem de visita das vítimas
5. **Planejamento**: A* e BFS calculam caminhos entre vítimas
6. **Execução**: Rescuer segue o plano calculado e retorna à base

#### Integração dos Algoritmos
- **Sistema Híbrido**: Combina algoritmos clássicos de IA (A*, BFS, K-Means, Genético) com técnicas modernas de machine learning
- **Otimização Multi-objetivo**: Maximiza número de vítimas resgatadas minimizando tempo e distância
- **Adaptação Dinâmica**: Algoritmos se ajustam conforme características do ambiente e número de vítimas
- **Robustez**: Fallbacks implementados para casos onde modelos ML não estão disponíveis

### 3.7 Explicabilidade
- **Processo deliberativo**: Implementação inspirada no artigo "Why Bad Coffee?" para justificar decisões de ordem de resgate.
- **Modelos explicáveis**: LIME/SHAP aplicados aos modelos vencedores para explicar decisões em instâncias de cada classe.
- **Saídas**: `lime_analysis.txt`, `cart_shap_summary.png` em `mas/models/`

---

## 4. Estrutura dos Diretórios e Arquivos

- `mas/` – Código-fonte principal dos agentes e lógica do sistema
- `mas/models/` – Modelos treinados, relatórios e explicações
- `mas/clusters/` – Saídas de agrupamento e sequenciamento (clusters e seqs)
- `datasets/` – Datasets de treinamento, validação e teste
- `tools/` – Scripts auxiliares para geração de dados e análise de resultados
- `Results_*/` – Resultados de execuções específicas
- `simulation_logs/` – Logs de simulação

---

## 5. Resultados e Análise

### 5.1 Exploração
- Estratégia, algoritmos utilizados e comparação com baseline (ver arquivos de resultados e gráficos em `Results_*/`)

### 5.2 Agrupamento
- Estratégia de clustering, análise dos grupos formados (ver arquivos `cluster*.txt`)

### 5.3 Sequenciamento
- Estratégia de otimização, análise das sequências geradas (ver arquivos `seq*.txt` e imagens de visualização)

### 5.4 Socorro
- Estratégia de execução, análise dos resultados comparados ao baseline

### 5.5 Classificador e Regressor
- Descrição dos pré-processamentos, validação cruzada, métricas (precisão, recall, f-measure, acurácia para classificadores; RMSE para regressores)
- Comparação das melhores configurações de cada algoritmo (ver relatórios em `mas/models/`)
- Justificativa da escolha final baseada em viés/variância (under/overfitting)

### 5.6 Explicabilidade
- Explicações do processo deliberativo e dos modelos (ver arquivos de explicação em `mas/models/`)

---

## 6. Ética
- Discussão sobre possíveis vieses, neutralidade e impactos sociais do sistema.

---

## 7. Como Reproduzir os Resultados
1. Instale as dependências: `pip install -r requirements.txt`
2. Execute os scripts de treinamento e simulação conforme desejado.
3. Analise os arquivos de saída gerados nos diretórios de resultados.

---

## 8. Referências
- Manual do VictimSim2 (`manuals/`)
- Artigo: Winikoff, M. et al. "Why Bad Coffee? Explaining BDI Agent Behaviour with Valuings". IJCAI 2022.

---

## 9. Contato
Dúvidas e sugestões: Matheus Vinicius Passos de Santana (santana.2003@alunos.utfpr.edu.br) e João Pedro Castilho Cardoso
