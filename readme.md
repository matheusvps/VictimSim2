# VictimSim2 – Sistema Multiagente para Resgate de Vítimas

## Descrição Geral
Este projeto implementa um Sistema Multiagente (SMA) para simulação de resgate de vítimas em cenários de catástrofes naturais, desastres ou grandes acidentes, utilizando o simulador VictimSim2. O sistema é composto por dois grupos de agentes artificiais (robôs terrestres): exploradores e socorristas, que atuam de forma sequencial para localizar e socorrer vítimas em um ambiente dinâmico e parcialmente desconhecido.

---

## 1. Arquitetura do Sistema

### 1.1 Fluxo Geral dos Agentes

O sistema opera em duas fases principais:

**Fase 1 - Exploração (4 Exploradores)**
- **Exploradores**: Localizam vítimas e constroem mapas parciais do ambiente, considerando restrições de bateria e obstáculos.
- **Sincronização**: Unificação dos mapas individuais em um único mapa global, compartilhado com os socorristas.

**Fase 2 - Resgate (4 Socorristas)**
- **Socorristas**: Recebem o mapa unificado, agrupam as vítimas, definem sequências de resgate otimizadas e realizam o socorro, retornando à base antes do fim da bateria.

O sistema integra algoritmos de busca, clustering, otimização (AG ou busca local), classificação e regressão, além de técnicas de explicabilidade (LIME/SHAP).

### 1.2 Configuração dos Agentes

#### Exploradores (4 agentes)
- **Configuração**: `explorer_1_config.txt` a `explorer_4_config.txt` em `cfg_1/` ou `cfg_2/`
- **Direções iniciais**: Cada explorador inicia em uma direção específica (0: direita, 2: baixo, 4: esquerda, 6: cima)
- **Regiões de exploração**: Podem ser definidas por coordenadas Y mínimas e máximas para distribuição do trabalho
- **Tempo limite**: 5000 unidades (configurável via `TLIM`)
- **Custos**: Movimento linear (1.0), diagonal (1.5), leitura (2.0), primeiros socorros (1.0)

#### Socorristas (4 agentes)
- **Configuração**: `rescuer_1_config.txt` a `rescuer_4_config.txt` em `cfg_1/` ou `cfg_2/`
- **Master Rescuer**: O primeiro socorrista (`rescuer_1`) é responsável por unificar os mapas dos exploradores
- **Tempo limite**: 1000 unidades (configurável via `TLIM`)
- **Custos**: Mesmos custos dos exploradores
- **Clusters**: Cada socorrista recebe um cluster específico de vítimas para resgatar

---

## 2. Execução do Sistema

### 2.1 Cenários Disponíveis

O sistema suporta múltiplos cenários com diferentes configurações:

#### Cenários Padrão
- **data_10v_12X12**: 10 vítimas em grid 12x12 (teste simples)
- **data_42v_20x20**: 42 vítimas em grid 20x20 (teste médio)
- **data_132v_100x80**: 132 vítimas em grid 100x80
- **data_225v_100x80**: 225 vítimas em grid 100x80
- **data_300v_90x90**: 300 vítimas em grid 90x90 (cenário de referência)
- **data_320v_90x90**: 320 vítimas em grid 90x90
- **data_400v_90x90**: 400 vítimas em grid 90x90
- **data_408v_94x94**: 408 vítimas em grid 94x94
- **data_4000v**: Dataset apenas com sinais vitais (para treinamento ML)

#### Configurações de Agentes
- **cfg_1**: Configuração padrão dos agentes
- **cfg_2**: Configuração alternativa dos agentes

### 2.2 Como Executar

#### Execução Individual
```bash
# Executar um dataset específico
python mas/main.py datasets/data_300v_90x90 cfg_1

# Executar com configuração alternativa
python mas/main.py datasets/data_408v_94x94 cfg_2
```

#### Execução de Múltiplos Datasets
```bash
# Executar todos os datasets
python run_all_datasets.py

# O script permite escolher datasets específicos ou executar todos
```

#### Treinamento de Modelos ML
```bash
# Treinar classificadores e regressores
python train_classifier.py

# Testar modelos treinados
python test_ml_models.py
```

### 2.3 Estrutura de Resultados

Os resultados são organizados em diretórios específicos:
- **Results_300v_90x90_cfg_1/**: Resultados do cenário de referência
- **Results_408v_94x94/**: Resultados de cenários específicos
- **simulation_logs/**: Logs detalhados de todas as simulações
- **mas/clusters/**: Clusters e sequências gerados durante a execução

---

## 3. Componentes e Algoritmos

### 3.1 Exploração - Fluxo Detalhado

O explorador implementa um **algoritmo híbrido inteligente** que combina múltiplas técnicas para maximizar a eficiência na descoberta de vítimas:

#### Inicialização dos Exploradores
```python
# Cada explorador é inicializado com:
- Direção inicial específica (0, 2, 4, 6)
- Região de exploração (coordenadas Y mín/máx)
- Mapa individual para construção
- Fronteira de exploração inicializada com posições adjacentes
```

#### Algoritmos Utilizados:

**1. Busca em Largura (BFS - Breadth-First Search)**
- **Exploração sistemática**: Garante que todas as células acessíveis sejam visitadas
- **Cálculo de caminhos**: Encontra o caminho mais curto para posições específicas
- **Navegação para a base**: Calcula o caminho de volta otimizado
- **Implementação**: Utiliza `deque` para fronteira e `set` para posições visitadas

**2. Algoritmo de Mapa de Calor (Heat Map)**
- **Priorização inteligente**: Direciona a exploração para áreas com maior probabilidade de vítimas
- **Atualização dinâmica**: 
  - Células com vítimas recebem pontuação +2.0
  - Células vizinhas recebem +0.5
  - Células sem vítimas perdem 0.1 pontos
  - Células vizinhas sem vítimas perdem 0.05
- **Adaptação contínua**: O mapa se ajusta conforme novas vítimas são descobertas

**3. Algoritmo de Detecção de Clusters**
- **Identificação de padrões**: Agrupa vítimas próximas para focar a exploração
- **Busca conectada**: Utiliza BFS para encontrar vítimas adjacentes
- **Estratégia de concentração**: Prioriza exploração em áreas com múltiplas vítimas

**4. Algoritmo de Priorização Inteligente**
- **Ordenação da fronteira**: Posições são ordenadas pelo valor do heat map
- **Cálculo de score**: Combina heat map com distância (score = heat - distância × 0.1)
- **Seleção adaptativa**: Escolhe a próxima posição baseada em múltiplos critérios

#### Fluxo de Decisão do Explorador
```python
def get_next_position(self):
    if not self.current_path:
        if not self.frontier:
            # Procura células não exploradas com maior probabilidade
            next_pos = self.find_best_unexplored_cell()
            if next_pos:
                self.current_path = self.calculate_path_to_position(next_pos)
                return self.current_path[0] if self.current_path else None
            return None

        # Ordena a fronteira pelo heat map
        frontier_list = list(self.frontier)
        frontier_list.sort(key=lambda x: self.heat_map.get((x[0], x[1]), 0), reverse=True)
        self.frontier = deque(frontier_list)

        next_x, next_y, path = self.frontier.popleft()
        self.visited.add((next_x, next_y))
        self.current_path = path
        return self.current_path[0]

    return self.current_path.pop(0)
```

#### Características do Algoritmo Híbrido:

- **Exploração Sistemática**: BFS garante cobertura completa do ambiente
- **Priorização Inteligente**: Heat map direciona para áreas promissoras
- **Detecção de Padrões**: Clusters identificam concentrações de vítimas
- **Navegação Eficiente**: Cálculo de caminhos otimizados para qualquer posição
- **Adaptação Dinâmica**: O algoritmo se ajusta conforme encontra vítimas
- **Retorno Garantido**: Cada explorador retorna à base antes do fim da bateria
- **Unificação de Mapas**: Mapas individuais são consolidados em um mapa global

### 3.2 Resgate - Fluxo Detalhado

#### Inicialização dos Socorristas
```python
# Master Rescuer (rescuer_1) é responsável por:
- Receber mapas de todos os exploradores
- Unificar informações de vítimas
- Distribuir clusters entre os socorristas

# Cada Rescuer recebe:
- Cluster específico de vítimas
- Mapa unificado do ambiente
- Tempo limite para resgate
```

#### Fase 1: Sincronização e Unificação
```python
def sync_explorers(self, explorer_map, victims):
    # Unifica mapas dos exploradores
    # Consolida informações de vítimas
    # Distribui clusters entre socorristas
```

#### Fase 2: Agrupamento de Vítimas (Clustering)

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

#### Fase 3: Classificação de Severidade

#### Machine Learning para Classificação de Severidade
- **Função**: `predict_severity_and_class()` no agente Rescuer
- **Propósito**: Classifica a severidade das vítimas usando modelo treinado
- **Características**:
  - Carrega classificador pré-treinado (pickle)
  - Usa scaler para normalização dos dados
  - Prediz classe de severidade (1=crítico, 2=instável, 3=potencialmente estável, 4=estável)
  - Fallback para valores aleatórios se modelo não carregar
  - Baseado em sinais vitais: pSist, pDiast, qPA, pulso, resp

#### Fase 4: Sequenciamento de Resgate

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

#### Fase 5: Planejamento de Caminhos

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

#### Fase 6: Execução do Socorro

Cada socorrista segue a sequência definida, socorrendo o máximo de vítimas possível e retornando à base antes do fim da bateria.

### 3.3 Fluxo Completo de Execução

#### Sequência de Processamento
1. **Inicialização**: 4 exploradores são criados com direções iniciais específicas
2. **Exploração Paralela**: Cada explorador mapeia sua região usando BFS e heat map
3. **Sincronização**: Master rescuer unifica mapas de todos os exploradores
4. **Clustering**: K-Means agrupa vítimas por proximidade geográfica
5. **Classificação**: ML classifica severidade das vítimas usando modelo treinado
6. **Distribuição**: Cada cluster é atribuído a um socorrista específico
7. **Sequenciamento**: Algoritmo genético otimiza ordem de visita das vítimas
8. **Planejamento**: A* e BFS calculam caminhos entre vítimas
9. **Execução Paralela**: Cada rescuer segue seu plano calculado e retorna à base

#### Integração dos Algoritmos
- **Sistema Híbrido**: Combina algoritmos clássicos de IA (A*, BFS, K-Means, Genético) com técnicas modernas de machine learning
- **Otimização Multi-objetivo**: Maximiza número de vítimas resgatadas minimizando tempo e distância
- **Adaptação Dinâmica**: Algoritmos se ajustam conforme características do ambiente e número de vítimas
- **Robustez**: Fallbacks implementados para casos onde modelos ML não estão disponíveis

### 3.4 Classificação e Regressão

#### Modelos Implementados
- **Classificador**: Estima a classe de gravidade da vítima (1=crítico, 2=instável, 3=potencialmente estável, 4=estável).
- **Regressor**: Estima o valor contínuo de gravidade [0, 100].
- **Modelos testados**: CART e Redes Neurais (MLP), com 3 configurações cada.
- **Pré-processamento**: Idêntico para ambos os métodos, sem uso de variáveis proibidas.
- **Validação cruzada**: Utilizada para avaliação robusta dos modelos.
- **Testes cegos**: Realizados com o dataset de 800 vítimas.
- **Arquivos de modelos**: Em `mas/models/` (ex: `cart_classifier.pkl`, `mlp_regressor.pkl`)
- **Relatórios e comparações**: Em `mas/models/` (ex: `algorithm_comparison_report.txt`, `prediction_report.txt`)

### 3.5 Explicabilidade
- **Processo deliberativo**: Implementação inspirada no artigo "Why Bad Coffee?" para justificar decisões de ordem de resgate.
- **Modelos explicáveis**: LIME/SHAP aplicados aos modelos vencedores para explicar decisões em instâncias de cada classe.
- **Saídas**: `lime_analysis.txt`, `cart_shap_summary.png` em `mas/models/`

### 3.6 Treinamento dos Modelos de Classificação e Regressão

O treinamento dos modelos de Machine Learning é realizado por meio do script `train_classifier.py`, que executa todas as etapas necessárias para gerar classificadores e regressores robustos para o sistema multiagente.

### **Etapas do Treinamento**
1. **Carregamento dos Dados**
   - Utiliza datasets localizados em `datasets/` (ex: `data_4000v/env_vital_signals.txt`).
   - Sinais vitais utilizados: `pSist`, `pDiast`, `qPA`, `pulso`, `resp`.
   - Variáveis alvo: `label` (classe de gravidade) para classificação e `grav` (valor contínuo) para regressão.

2. **Pré-processamento**
   - Normalização dos dados com `StandardScaler`.
   - O pré-processamento é idêntico para todos os algoritmos concorrentes.
   - **Restrição importante:** Não é permitido usar a classe de gravidade como entrada do regressor, nem o valor de gravidade como entrada do classificador.

3. **Configuração dos Modelos**
   - Dois algoritmos testados: **CART (Árvore de Decisão)** e **Redes Neurais (MLP)**.
   - Para cada algoritmo, são avaliadas 3 configurações diferentes de hiperparâmetros.

4. **Validação Cruzada e Seleção**
   - Utiliza validação cruzada (K-Fold) para avaliar o desempenho de cada configuração.
   - Métricas salvas:
     - **Classificador:** acurácia, precisão, recall, f1-score.
     - **Regressor:** RMSE, MAE, R².
   - Seleção automática da melhor configuração de cada algoritmo.

5. **Comparação e Relatórios**
   - Geração de relatórios comparativos entre os melhores modelos de cada algoritmo.
   - Relatórios e gráficos salvos em `mas/models/`:
     - `classifier_comparison_report.txt`, `classifier_comparison.png`
     - `regressor_comparison_report.txt`, `regressor_comparison.png`
     - `prediction_report_comparison.txt`

6. **Explicabilidade**
   - Aplicação de LIME e SHAP para explicar as decisões dos modelos vencedores.
   - Saídas de explicabilidade: `lime_analysis.txt`, `cart_shap_summary.png`.

7. **Salvamento dos Modelos**
   - Modelos e scalers salvos em formato pickle (`.pkl`) em `mas/models/`:
     - `cart_classifier.pkl`, `mlp_classifier.pkl`, `cart_regressor.pkl`, `mlp_regressor.pkl`
     - Scalers correspondentes: `*_scaler.pkl`

### **Integração com o Rescuer**
- O agente Rescuer carrega automaticamente o classificador e o scaler treinados ao iniciar a simulação.
- As predições de classe de gravidade são feitas usando os sinais vitais das vítimas encontradas.
- O valor de gravidade pode ser integrado futuramente usando o regressor salvo.

### **Arquivos Gerados**
- Modelos treinados: `mas/models/*.pkl`
- Relatórios e explicações: `mas/models/*.txt`, `mas/models/*.png`

Para mais detalhes, consulte o script `train_classifier.py` e os relatórios gerados após o treinamento.

### **Arquivos Gerados em `mas/models/`**

A execução do treinamento gera os seguintes arquivos em `mas/models/`:

#### **Modelos Treinados e Scalers**
- `cart_classifier.pkl`, `mlp_classifier.pkl`, `victim_classifier.pkl`, `perceptron_classifier.pkl`: Modelos de classificação treinados (CART, MLP, Perceptron).
- `cart_regressor.pkl`, `mlp_regressor.pkl`: Modelos de regressão treinados.
- `*_scaler.pkl`: Scalers de normalização usados para cada modelo.

#### **Relatórios de Resultados**
- `classifier_comparison_report.txt`, `regressor_comparison_report.txt`, `algorithm_comparison_report.txt`: Relatórios comparativos entre algoritmos e configurações.
- `prediction_report_comparison.txt`, `prediction_report.txt`: Relatórios detalhados das predições de todos os modelos para cada vítima.

#### **Gráficos e Visualizações**
- `classifier_comparison.png`, `regressor_comparison.png`, `algorithm_comparison.png`: Gráficos comparativos de desempenho dos modelos.
- `cart_feature_importance.png`: Importância das features para o classificador CART.
- `cart_shap_summary.png`: Gráfico de explicabilidade SHAP para o classificador CART.

#### **Explicabilidade**
- `lime_analysis.txt`: Explicações LIME para instâncias de cada classe.

Esses arquivos documentam todo o processo de treinamento, validação, comparação e explicação dos modelos utilizados pelo sistema. Consulte-os para análise detalhada dos resultados e para integração dos modelos ao sistema multiagente.

### **4. Análise dos Resultados dos Modelos de Classificação**

#### **4.1 Avaliação Quantitativa**

O treinamento dos classificadores CART e Redes Neurais (MLP) foi realizado com três configurações para cada algoritmo. As métricas de desempenho (precisão, recall, f-measure e acurácia) para cada configuração estão detalhadas em `mas/models/classifier_config_results.txt`. Os melhores resultados foram:

- **CART**: Acurácia até 0.9287, f-measure 0.9288
- **MLP**: Acurácia até 0.9225, f-measure 0.9228

Esses valores indicam que ambos os modelos apresentam excelente desempenho, com leve vantagem para o CART nas configurações mais agressivas.

#### **4.2 Análise de Explicabilidade (LIME/SHAP)**

Foram geradas análises de explicabilidade utilizando LIME e SHAP para os modelos treinados:

- **LIME**: Permitiu identificar quais sinais vitais mais influenciam a decisão do classificador para cada vítima individualmente. Observou-se que variáveis como pulso, respiração e pressão arterial têm maior impacto na classificação.
- **SHAP**: A análise global de importância de features mostrou que `pulso` e `qPA` são as variáveis mais relevantes para o modelo CART, seguidas por `resp` e `pSist`.

Essas ferramentas aumentam a confiança no modelo, permitindo interpretar e justificar as decisões dos classificadores, especialmente em cenários críticos como o resgate de vítimas.

#### **4.3 Matriz de Confusão**

A matriz de confusão dos melhores modelos mostra que a maioria das vítimas é corretamente classificada em suas respectivas categorias de gravidade. Os poucos erros ocorrem principalmente entre classes vizinhas (ex: INSTÁVEL vs. POTENCIALMENTE ESTÁVEL), o que é esperado devido à proximidade clínica dos casos.

#### **4.4 Justificativa da Escolha do Modelo (Viés e Variância)**

Durante o treinamento, foi observada a seguinte relação entre as configurações:

- **Configurações conservadoras** (menor profundidade/camadas): apresentaram maior viés (underfitting), com acurácia e f-measure mais baixas.
- **Configurações agressivas** (maior profundidade/camadas): apresentaram menor viés e variância controlada, sem sinais de overfitting (a acurácia de teste se manteve próxima à de validação cruzada).

**CART** foi escolhido como modelo principal devido à sua leve superioridade em desempenho, maior interpretabilidade e robustez frente a variações nos dados. O MLP também apresentou bom desempenho e pode ser utilizado como alternativa, especialmente em cenários onde a flexibilidade do modelo é desejada.

**Resumo da escolha:**
- **CART**: Preferido por ser mais interpretável, robusto e apresentar leve vantagem nas métricas.
- **MLP**: Útil como alternativa, especialmente se houver aumento de complexidade dos dados.

---

## 4. Estrutura dos Diretórios e Arquivos

- `mas/` – Código-fonte principal dos agentes e lógica do sistema
- `mas/models/` – Modelos treinados, relatórios e explicações
- `mas/clusters/` – Saídas de agrupamento e sequenciamento (clusters e seqs)
- `datasets/` – Datasets de treinamento, validação e teste
- `tools/` – Scripts auxiliares para geração de dados e análise de resultados
- `Results_*/` – Resultados de execuções específicas
- `simulation_logs/` – Logs de simulação
- `cfg_1/` e `cfg_2/` – Configurações dos agentes

---

## 5. Resultados e Análise

### 5.1 Exploração

```python
# Estratégia de exploração híbrida: BFS + Heat Map + Clusters
def get_next_position(self):
    if not self.current_path:
        if not self.frontier:
            next_pos = self.find_best_unexplored_cell()
            if next_pos:
                self.current_path = self.calculate_path_to_position(next_pos)
                return self.current_path[0] if self.current_path else None
            return None
        # Ordena a fronteira pelo heat map
        frontier_list = list(self.frontier)
        frontier_list.sort(key=lambda x: self.heat_map.get((x[0], x[1]), 0), reverse=True)
        self.frontier = deque(frontier_list)
        next_x, next_y, path = self.frontier.popleft()
        if (next_x, next_y) in self.visited:
            return self.get_next_position()
        self.visited.add((next_x, next_y))
        self.current_path = path
        return self.current_path[0]
    return self.current_path.pop(0)
```
*Esse trecho mostra a priorização inteligente da exploração, combinando BFS e mapa de calor.*

---

### 5.2 Agrupamento

```python
# Agrupamento de vítimas usando K-Means adaptado
def cluster_victims(self):
    num_clusters = min(4, len(self.victims))
    centroids = dict(random.sample(list(self.victims.items()), num_clusters))
    for i in range(4):  # Máximo 4 iterações
        clusters = [{} for _ in range(num_clusters)]
        # Atribuição das vítimas ao cluster mais próximo
        for key, values in self.victims.items():
            x, y = values[0]
            min_key = min(centroids, key=lambda c: math.dist((x, y), centroids[c][0]))
            idx = list(centroids.keys()).index(min_key)
            clusters[idx][key] = values
        # Atualização dos centróides
        for idx, cluster in enumerate(clusters):
            if cluster:
                mean_x = sum(v[0][0] for v in cluster.values()) / len(cluster)
                mean_y = sum(v[0][1] for v in cluster.values()) / len(cluster)
                centroids[list(centroids.keys())[idx]] = ((mean_x, mean_y), centroids[list(centroids.keys())[idx]][1])
    return clusters
```
*Exemplo de agrupamento dinâmico de vítimas. Resultados podem ser visualizados nos arquivos `cluster*.txt`.*

---

### 5.3 Sequenciamento

```python
# Sequenciamento de resgate com Algoritmo Genético
def genetic_algorithm(indexed_victims, distance_matrix, pop_size, generations, cx_prob, mut_prob):
    # Inicialização da população
    # Seleção por torneio, crossover e mutação
    # Avaliação por fitness (menor distância total)
    # Retorna a melhor sequência encontrada
    ...
```
*O sequenciamento otimiza a ordem de visita das vítimas. Visualize as sequências e clusters com:*
```sh
python tools/results/plot_clusters_and_seq.py
```
*Inclua a imagem gerada (`results_graphics.png` ou similar) na apresentação.*

---

### 5.4 Socorro

```python
# Execução do socorro com A* para trajetórias ótimas
def a_star(map, coord_start, coord_goal, min_difficulty):
    # Busca o menor caminho considerando custos e obstáculos
    ...
```
*O socorrista segue a sequência calculada, usando A* para cada trajeto entre vítimas e retorno à base.*

---

### 5.5 Classificador e Regressor

```python
# Treinamento do classificador CART (exemplo de configuração)
cart_configs = [
    {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'criterion': 'gini'},
    {'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy'},
    {'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'}
]
# Treinamento e validação cruzada
cart_classifier = DecisionTreeClassifier(**cart_configs[0])
cv_scores = cross_val_score(cart_classifier, X_train_scaled, y_train, cv=5)
```
*Resultados e métricas (precisão, recall, f1, acurácia) estão em `mas/models/classifier_comparison_report.txt`.*

```python
# Treinamento do regressor CART (exemplo de configuração)
cart_regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, criterion='squared_error')
cv_scores = cross_val_score(cart_regressor, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
```
*Resultados de RMSE, MAE, R² em `mas/models/regressor_comparison_report.txt`.*

---

### 5.6 Explicabilidade

```python
# Geração de explicações LIME e SHAP
generate_explainability_analysis(cart_results, mlp_results, X, y, features, class_names, output_dir)
```
*Veja exemplos em `mas/models/lime_analysis.txt` e `cart_shap_summary.png`.*

---

### 5.7 Sistema Multiagente

```python
class AbstAgent(ABC):
    def deliberate(self) -> bool:
        """Escolha da próxima ação. Chamado a cada ciclo de raciocínio."""
        pass
```
*Todos os agentes (explorador, socorrista) herdam de `AbstAgent`, integrando suas ações no ambiente.*

---

### 6. Ética

> **Discussão:**  
> O sistema pode apresentar viés caso os dados de treinamento não representem adequadamente todos os tipos de vítimas. A neutralidade depende da qualidade dos dados e da transparência dos modelos. Situações não previstas (ex: vítimas com sinais vitais atípicos) podem afetar a equidade do resgate.

---

### 5.8 Justificativa de escolha dos modelos

> **Justificativa:**  
> A escolha entre CART e MLP é feita com base em análise de underfitting/overfitting (viés/variância), usando validação cruzada e testes cegos (dataset de 800 vítimas). Veja detalhes nos relatórios de comparação.

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

## Visualização dos Resultados

Para visualizar os resultados detalhados dos exploradores, resgatadores e gerais, utilize o script `plot_results.py`.

### Como executar

```bash
python plot_results.py
```

### Exemplo de código para plotar os resultados

```python
import matplotlib.pyplot as plt
import numpy as np

# Exemplo: Gráfico de vítimas encontradas por cada explorador
exploradores = {
    'EXPL_1': {'Críticas': 6, 'Instáveis': 15, 'Pot. Instáveis': 7, 'Estáveis': 4},
    'EXPL_2': {'Críticas': 8, 'Instáveis': 15, 'Pot. Instáveis': 4, 'Estáveis': 2},
    'EXPL_3': {'Críticas': 9, 'Instáveis': 12, 'Pot. Instáveis': 6, 'Estáveis': 1},
    'EXPL_4': {'Críticas': 15, 'Instáveis': 16, 'Pot. Instáveis': 7, 'Estáveis': 4},
}
labels = ['Críticas', 'Instáveis', 'Pot. Instáveis', 'Estáveis']
x = np.arange(len(labels))
bar_width = 0.18
fig, ax = plt.subplots(figsize=(10,6))
for i, (agente, dados) in enumerate(exploradores.items()):
    valores = [dados[k] for k in labels]
    ax.bar(x + i*bar_width - 1.5*bar_width, valores, bar_width, label=agente)
ax.set_ylabel('Vítimas encontradas')
ax.set_title('Vítimas encontradas por cada explorador')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()
```

O script completo já gera gráficos para:
- Vítimas encontradas por explorador (absoluto e percentual)
- Vítimas salvas por resgatador (absoluto e percentual)
- Comparativo geral (acumulado) de vítimas no ambiente, encontradas e salvas
- Gravidade total das vítimas

Adapte os exemplos conforme necessário para sua análise!
