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
- Algoritmo de busca (ex: BFS, DFS, A*) para exploração do ambiente.
- Cada explorador retorna à base antes do fim da bateria para garantir que os dados coletados não sejam perdidos.
- Unificação dos mapas individuais em um único mapa global.

### 3.2 Agrupamento de Vítimas (Clustering)
- Algoritmo de clustering (ex: K-Means, DBSCAN) para dividir as vítimas em grupos, um para cada socorrista.
- Arquivos gerados: `cluster1.txt`, `cluster2.txt`, ... (em `mas/clusters/` ou diretórios de resultados)
- Formato: `id, x, y, grav, classe`

### 3.3 Sequenciamento de Resgate
- Algoritmo Genético (AG) ou busca local (ex: Têmpera Simulada) para definir a ordem de resgate das vítimas de cada grupo.
- Restrições de bateria e tempo consideradas.
- Arquivos gerados: `seq1.txt`, `seq2.txt`, ... (em `mas/clusters/` ou diretórios de resultados)
- Formato: `id, x, y, grav, classe`

### 3.4 Execução do Socorro
- Cada socorrista segue a sequência definida, socorrendo o máximo de vítimas possível e retornando à base antes do fim da bateria.
- Algoritmo de busca de caminho (ex: A*) para deslocamento entre vítimas.

### 3.5 Classificação e Regressão
- **Classificador**: Estima a classe de gravidade da vítima (1=crítico, 2=instável, 3=potencialmente estável, 4=estável).
- **Regressor**: Estima o valor contínuo de gravidade [0, 100].
- **Modelos testados**: CART e Redes Neurais (MLP), com 3 configurações cada.
- **Pré-processamento**: Idêntico para ambos os métodos, sem uso de variáveis proibidas.
- **Validação cruzada**: Utilizada para avaliação robusta dos modelos.
- **Testes cegos**: Realizados com o dataset de 800 vítimas.
- **Arquivos de modelos**: Em `mas/models/` (ex: `cart_classifier.pkl`, `mlp_regressor.pkl`)
- **Relatórios e comparações**: Em `mas/models/` (ex: `algorithm_comparison_report.txt`, `prediction_report.txt`)

### 3.6 Explicabilidade
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
