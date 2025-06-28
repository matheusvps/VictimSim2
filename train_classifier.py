#!/usr/bin/env python3
"""
Script para treinar classificadores e regressores usando o dataset 4000v
e salvar os modelos treinados para uso posterior pelos rescuers.

Implementa dois algoritmos diferentes com 3 configurações cada:
1. CART (Decision Tree) - Árvore de Decisão
2. Redes Neurais (MLP) - Rede Neural Multilayer Perceptron

Para cada algoritmo, testa 3 configurações diferentes e seleciona a melhor.
Inclui validação cruzada, explicabilidade (LIME/SHAP) e comparação entre algoritmos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Para explicabilidade
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME não disponível. Instale com: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP não disponível. Instale com: pip install shap")

def load_vital_signals_data(file_path):
    """
    Carrega os dados de sinais vitais do arquivo env_vital_signals.txt
    
    Formato esperado: i, pSist, pDiast, qPA, pulso, resp, grav, label
    """
    print(f"Carregando dados de sinais vitais de: {file_path}")
    
    # Lê o arquivo de sinais vitais
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(x.strip()) for x in line.split(',')]
                data.append(values)
    
    # Converte para DataFrame
    columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'label']
    df = pd.DataFrame(data, columns=columns)
    
    print(f"Dados carregados: {len(df)} registros")
    print(f"Distribuição das classes:")
    print(df['label'].value_counts().sort_index())
    
    return df

def prepare_features(df):
    """
    Prepara as features para treinamento (excluindo grav e label)
    """
    # Features: pSist, pDiast, qPA, pulso, resp
    features = ['pSist', 'pDiast', 'qPA', 'pulso', 'resp']
    X = df[features].values
    y_class = df['label'].values
    y_reg = df['grav'].values
    
    return X, y_class, y_reg, features

def train_cart_classifier_configs(X, y, class_names, cv_folds=5):
    """
    Treina classificadores CART com 3 configurações diferentes
    """
    print("\n" + "="*60)
    print("TREINANDO CLASSIFICADORES CART - 3 CONFIGURAÇÕES")
    print("="*60)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurações CART para teste
    cart_configs = [
        {
            'name': 'Config 1 - Conservadora',
            'params': {
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'criterion': 'gini'
            }
        },
        {
            'name': 'Config 2 - Moderada',
            'params': {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'criterion': 'entropy'
            }
        },
        {
            'name': 'Config 3 - Agressiva',
            'params': {
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'criterion': 'gini'
            }
        }
    ]
    
    best_config = None
    best_score = 0
    all_results = []
    
    for i, config in enumerate(cart_configs):
        print(f"\n--- Testando {config['name']} ---")
        
        # Configura o classificador CART
        cart_classifier = DecisionTreeClassifier(**config['params'])
        
        # Validação cruzada
        cv_scores = cross_val_score(
            cart_classifier, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        print(f"Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Treina o modelo final
        cart_classifier.fit(X_train_scaled, y_train)
        
        # Avalia o modelo no conjunto de teste
        y_pred_test = cart_classifier.predict(X_test_scaled)
        
        # Métricas de avaliação
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"Acurácia (teste): {accuracy:.4f}")
        print(f"F1-Score (teste): {f1:.4f}")
        
        result = {
            'config_name': config['name'],
            'config_params': config['params'],
            'classifier': cart_classifier,
            'scaler': scaler,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
        
        all_results.append(result)
        
        # Atualiza o melhor se necessário
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_config = result
    
    print(f"\n=== MELHOR CONFIGURAÇÃO CART ===")
    print(f"Configuração: {best_config['config_name']}")
    print(f"CV Score: {best_config['cv_mean']:.4f} (+/- {best_config['cv_std'] * 2:.4f})")
    print(f"Acurácia (teste): {best_config['metrics']['accuracy']:.4f}")
    
    # Faz predições para todos os dados com o melhor modelo
    X_scaled = scaler.transform(X)
    y_pred_all = best_config['classifier'].predict(X_scaled)
    best_config['y_pred_all'] = y_pred_all
    
    return best_config, all_results

def train_mlp_classifier_configs(X, y, class_names, cv_folds=5):
    """
    Treina classificadores MLP com 3 configurações diferentes
    """
    print("\n" + "="*60)
    print("TREINANDO CLASSIFICADORES MLP - 3 CONFIGURAÇÕES")
    print("="*60)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurações MLP para teste
    mlp_configs = [
        {
            'name': 'Config 1 - Simples',
            'params': {
                'hidden_layer_sizes': (50,),
                'max_iter': 500,
                'learning_rate_init': 0.1,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        },
        {
            'name': 'Config 2 - Moderada',
            'params': {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                'learning_rate_init': 0.01,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        },
        {
            'name': 'Config 3 - Complexa',
            'params': {
                'hidden_layer_sizes': (200, 100, 50),
                'max_iter': 1500,
                'learning_rate_init': 0.001,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        }
    ]
    
    best_config = None
    best_score = 0
    all_results = []
    
    for i, config in enumerate(mlp_configs):
        print(f"\n--- Testando {config['name']} ---")
        
        # Configura o classificador MLP
        mlp_classifier = MLPClassifier(**config['params'])
        
        # Validação cruzada
        cv_scores = cross_val_score(
            mlp_classifier, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        print(f"Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Treina o modelo final
        mlp_classifier.fit(X_train_scaled, y_train)
        
        # Avalia o modelo no conjunto de teste
        y_pred_test = mlp_classifier.predict(X_test_scaled)
        
        # Métricas de avaliação
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted')
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        print(f"Acurácia (teste): {accuracy:.4f}")
        print(f"F1-Score (teste): {f1:.4f}")
        
        result = {
            'config_name': config['name'],
            'config_params': config['params'],
            'classifier': mlp_classifier,
            'scaler': scaler,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
        
        all_results.append(result)
        
        # Atualiza o melhor se necessário
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_config = result
    
    print(f"\n=== MELHOR CONFIGURAÇÃO MLP ===")
    print(f"Configuração: {best_config['config_name']}")
    print(f"CV Score: {best_config['cv_mean']:.4f} (+/- {best_config['cv_std'] * 2:.4f})")
    print(f"Acurácia (teste): {best_config['metrics']['accuracy']:.4f}")
    
    # Faz predições para todos os dados com o melhor modelo
    X_scaled = scaler.transform(X)
    y_pred_all = best_config['classifier'].predict(X_scaled)
    best_config['y_pred_all'] = y_pred_all
    
    return best_config, all_results

def train_cart_regressor_configs(X, y, cv_folds=5):
    """
    Treina regressores CART com 3 configurações diferentes
    """
    print("\n" + "="*60)
    print("TREINANDO REGRESSORES CART - 3 CONFIGURAÇÕES")
    print("="*60)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurações CART para teste
    cart_configs = [
        {
            'name': 'Config 1 - Conservadora',
            'params': {
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'criterion': 'squared_error'
            }
        },
        {
            'name': 'Config 2 - Moderada',
            'params': {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'criterion': 'absolute_error'
            }
        },
        {
            'name': 'Config 3 - Agressiva',
            'params': {
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'criterion': 'squared_error'
            }
        }
    ]
    
    best_config = None
    best_score = float('inf')  # Para regressão, menor é melhor
    all_results = []
    
    for i, config in enumerate(cart_configs):
        print(f"\n--- Testando {config['name']} ---")
        
        # Configura o regressor CART
        cart_regressor = DecisionTreeRegressor(**config['params'])
        
        # Validação cruzada
        cv_scores = cross_val_score(
            cart_regressor, X_train_scaled, y_train, 
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"RMSE médio (CV): {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        
        # Treina o modelo final
        cart_regressor.fit(X_train_scaled, y_train)
        
        # Avalia o modelo no conjunto de teste
        y_pred_test = cart_regressor.predict(X_test_scaled)
        
        # Métricas de avaliação
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print(f"RMSE (teste): {rmse:.4f}")
        print(f"R² (teste): {r2:.4f}")
        
        result = {
            'config_name': config['name'],
            'config_params': config['params'],
            'regressor': cart_regressor,
            'scaler': scaler,
            'cv_scores': cv_rmse,
            'cv_mean': cv_rmse.mean(),
            'cv_std': cv_rmse.std(),
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }
        
        all_results.append(result)
        
        # Atualiza o melhor se necessário
        if cv_rmse.mean() < best_score:
            best_score = cv_rmse.mean()
            best_config = result
    
    print(f"\n=== MELHOR CONFIGURAÇÃO CART REGRESSOR ===")
    print(f"Configuração: {best_config['config_name']}")
    print(f"CV RMSE: {best_config['cv_mean']:.4f} (+/- {best_config['cv_std'] * 2:.4f})")
    print(f"RMSE (teste): {best_config['metrics']['rmse']:.4f}")
    
    # Faz predições para todos os dados com o melhor modelo
    X_scaled = scaler.transform(X)
    y_pred_all = best_config['regressor'].predict(X_scaled)
    best_config['y_pred_all'] = y_pred_all
    
    return best_config, all_results

def train_mlp_regressor_configs(X, y, cv_folds=5):
    """
    Treina regressores MLP com 3 configurações diferentes
    """
    print("\n" + "="*60)
    print("TREINANDO REGRESSORES MLP - 3 CONFIGURAÇÕES")
    print("="*60)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurações MLP para teste
    mlp_configs = [
        {
            'name': 'Config 1 - Simples',
            'params': {
                'hidden_layer_sizes': (50,),
                'max_iter': 500,
                'learning_rate_init': 0.1,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        },
        {
            'name': 'Config 2 - Moderada',
            'params': {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                'learning_rate_init': 0.01,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        },
        {
            'name': 'Config 3 - Complexa',
            'params': {
                'hidden_layer_sizes': (200, 100, 50),
                'max_iter': 1500,
                'learning_rate_init': 0.001,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        }
    ]
    
    best_config = None
    best_score = float('inf')  # Para regressão, menor é melhor
    all_results = []
    
    for i, config in enumerate(mlp_configs):
        print(f"\n--- Testando {config['name']} ---")
        
        # Configura o regressor MLP
        mlp_regressor = MLPRegressor(**config['params'])
        
        # Validação cruzada
        cv_scores = cross_val_score(
            mlp_regressor, X_train_scaled, y_train, 
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"RMSE médio (CV): {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        
        # Treina o modelo final
        mlp_regressor.fit(X_train_scaled, y_train)
        
        # Avalia o modelo no conjunto de teste
        y_pred_test = mlp_regressor.predict(X_test_scaled)
        
        # Métricas de avaliação
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print(f"RMSE (teste): {rmse:.4f}")
        print(f"R² (teste): {r2:.4f}")
        
        result = {
            'config_name': config['name'],
            'config_params': config['params'],
            'regressor': mlp_regressor,
            'scaler': scaler,
            'cv_scores': cv_rmse,
            'cv_mean': cv_rmse.mean(),
            'cv_std': cv_rmse.std(),
            'y_pred_test': y_pred_test,
            'y_test': y_test,
            'X_test_scaled': X_test_scaled,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        }
        
        all_results.append(result)
        
        # Atualiza o melhor se necessário
        if cv_rmse.mean() < best_score:
            best_score = cv_rmse.mean()
            best_config = result
    
    print(f"\n=== MELHOR CONFIGURAÇÃO MLP REGRESSOR ===")
    print(f"Configuração: {best_config['config_name']}")
    print(f"CV RMSE: {best_config['cv_mean']:.4f} (+/- {best_config['cv_std'] * 2:.4f})")
    print(f"RMSE (teste): {best_config['metrics']['rmse']:.4f}")
    
    # Faz predições para todos os dados com o melhor modelo
    X_scaled = scaler.transform(X)
    y_pred_all = best_config['regressor'].predict(X_scaled)
    best_config['y_pred_all'] = y_pred_all
    
    return best_config, all_results

def generate_explainability_analysis(cart_results, mlp_results, X, y, features, class_names, output_dir):
    """
    Gera análises de explicabilidade usando LIME e SHAP
    """
    print("\n" + "="*60)
    print("GERANDO ANÁLISES DE EXPLICABILIDADE")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Análise de importância de features (CART)
    print("\n1. Análise de Importância de Features (CART)")
    cart_classifier = cart_results['classifier']
    feature_importance = cart_classifier.feature_importances_
    
    # Gráfico de importância de features
    plt.figure(figsize=(10, 6))
    plt.bar(features, feature_importance)
    plt.title('Importância das Features - Classificador CART')
    plt.xlabel('Features')
    plt.ylabel('Importância')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cart_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Importância das features (CART):")
    for feature, importance in zip(features, feature_importance):
        print(f"  {feature}: {importance:.4f}")
    
    # Análise LIME (se disponível)
    if LIME_AVAILABLE:
        print("\n2. Análise LIME")
        try:
            # Cria o explainer LIME
            explainer = lime.lime_tabular.LimeTabularExplainer(
                cart_results['X_test_scaled'],
                feature_names=features,
                class_names=class_names,
                mode='classification'
            )
            
            # Analisa alguns exemplos
            lime_report_path = os.path.join(output_dir, 'lime_analysis.txt')
            with open(lime_report_path, 'w', encoding='utf-8') as f:
                f.write("=== ANÁLISE LIME - EXPLICABILIDADE ===\n\n")
                
                # Analisa 5 exemplos de cada classe
                for class_idx in range(1, 5):
                    class_mask = cart_results['y_test'] == class_idx
                    if class_mask.sum() > 0:
                        example_idx = np.where(class_mask)[0][0]
                        exp = explainer.explain_instance(
                            cart_results['X_test_scaled'][example_idx],
                            cart_results['classifier'].predict_proba,
                            num_features=len(features)
                        )
                        
                        f.write(f"Exemplo da classe {class_names[class_idx-1]}:\n")
                        f.write(f"Features mais importantes:\n")
                        for feature, weight in exp.as_list():
                            f.write(f"  {feature}: {weight:.4f}\n")
                        f.write("\n")
            
            print(f"Análise LIME salva em: {lime_report_path}")
            
        except Exception as e:
            print(f"Erro na análise LIME: {e}")
    
    # Análise SHAP (se disponível)
    if SHAP_AVAILABLE:
        print("\n3. Análise SHAP")
        try:
            # Para CART
            explainer = shap.TreeExplainer(cart_results['classifier'])
            shap_values = explainer.shap_values(cart_results['X_test_scaled'][:100])  # Primeiros 100 exemplos
            
            # Gráfico de resumo SHAP
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, cart_results['X_test_scaled'][:100], 
                            feature_names=features, class_names=class_names, show=False)
            plt.title('Análise SHAP - Classificador CART')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cart_shap_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Análise SHAP para CART concluída")
            
        except Exception as e:
            print(f"Erro na análise SHAP: {e}")

def compare_classifiers(cart_results, mlp_results, output_dir):
    """
    Compara os dois algoritmos classificadores e gera relatórios comparativos
    """
    print("\n" + "="*60)
    print("COMPARANDO CLASSIFICADORES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparação de métricas
    comparison_data = {
        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
        'CART': [
            cart_results['metrics']['accuracy'],
            cart_results['metrics']['precision'],
            cart_results['metrics']['recall'],
            cart_results['metrics']['f1']
        ],
        'MLP': [
            mlp_results['metrics']['accuracy'],
            mlp_results['metrics']['precision'],
            mlp_results['metrics']['recall'],
            mlp_results['metrics']['f1']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Gráfico de comparação
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Métricas comparativas
    plt.subplot(2, 2, 1)
    x = np.arange(len(comparison_data['Métrica']))
    width = 0.35
    
    plt.bar(x - width/2, comparison_data['CART'], width, label='CART', alpha=0.8)
    plt.bar(x + width/2, comparison_data['MLP'], width, label='MLP', alpha=0.8)
    
    plt.xlabel('Métricas')
    plt.ylabel('Score')
    plt.title('Comparação de Métricas - Classificadores')
    plt.xticks(x, comparison_data['Métrica'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Validação cruzada
    plt.subplot(2, 2, 2)
    cv_data = [cart_results['cv_scores'], mlp_results['cv_scores']]
    plt.boxplot(cv_data, labels=['CART', 'MLP'])
    plt.ylabel('Acurácia (CV)')
    plt.title('Validação Cruzada - Classificadores')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Matriz de confusão CART
    plt.subplot(2, 2, 3)
    cm_cart = confusion_matrix(cart_results['y_test'], cart_results['y_pred_test'])
    sns.heatmap(cm_cart, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - CART')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    # Subplot 4: Matriz de confusão MLP
    plt.subplot(2, 2, 4)
    cm_mlp = confusion_matrix(mlp_results['y_test'], mlp_results['y_pred_test'])
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Reds')
    plt.title('Matriz de Confusão - MLP')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classifier_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Relatório de comparação
    comparison_report_path = os.path.join(output_dir, 'classifier_comparison_report.txt')
    with open(comparison_report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE COMPARAÇÃO ENTRE CLASSIFICADORES ===\n\n")
        
        f.write("1. MÉTRICAS DE PERFORMANCE:\n")
        f.write("-" * 50 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("2. VALIDAÇÃO CRUZADA:\n")
        f.write("-" * 50 + "\n")
        f.write(f"CART - Média: {cart_results['cv_mean']:.4f}, Std: {cart_results['cv_std']:.4f}\n")
        f.write(f"MLP - Média: {mlp_results['cv_mean']:.4f}, Std: {mlp_results['cv_std']:.4f}\n\n")
        
        f.write("3. ANÁLISE COMPARATIVA:\n")
        f.write("-" * 50 + "\n")
        
        # Determina o melhor algoritmo para cada métrica
        best_accuracy = "CART" if cart_results['metrics']['accuracy'] > mlp_results['metrics']['accuracy'] else "MLP"
        best_precision = "CART" if cart_results['metrics']['precision'] > mlp_results['metrics']['precision'] else "MLP"
        best_recall = "CART" if cart_results['metrics']['recall'] > mlp_results['metrics']['recall'] else "MLP"
        best_f1 = "CART" if cart_results['metrics']['f1'] > mlp_results['metrics']['f1'] else "MLP"
        
        f.write(f"Melhor Acurácia: {best_accuracy}\n")
        f.write(f"Melhor Precisão: {best_precision}\n")
        f.write(f"Melhor Recall: {best_recall}\n")
        f.write(f"Melhor F1-Score: {best_f1}\n\n")
        
        f.write("4. RECOMENDAÇÕES:\n")
        f.write("-" * 50 + "\n")
        f.write("• CART é mais interpretável e pode capturar relações não-lineares\n")
        f.write("• MLP é mais flexível mas pode ser menos interpretável\n")
        f.write("• Para este problema específico, recomenda-se usar ambos os modelos\n")
        f.write("  e escolher baseado no contexto de uso (interpretabilidade vs flexibilidade)\n")
    
    print(f"Relatório de comparação salvo em: {comparison_report_path}")
    print(f"Gráficos de comparação salvos em: {output_dir}")

def compare_regressors(cart_results, mlp_results, output_dir):
    """
    Compara os dois algoritmos regressores e gera relatórios comparativos
    """
    print("\n" + "="*60)
    print("COMPARANDO REGRESSORES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparação de métricas
    comparison_data = {
        'Métrica': ['RMSE', 'MAE', 'R²'],
        'CART': [
            cart_results['metrics']['rmse'],
            cart_results['metrics']['mae'],
            cart_results['metrics']['r2']
        ],
        'MLP': [
            mlp_results['metrics']['rmse'],
            mlp_results['metrics']['mae'],
            mlp_results['metrics']['r2']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Gráfico de comparação
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Métricas comparativas (RMSE e MAE - menor é melhor)
    plt.subplot(2, 2, 1)
    x = np.arange(2)  # RMSE e MAE
    width = 0.35
    
    plt.bar(x - width/2, comparison_data['CART'][:2], width, label='CART', alpha=0.8)
    plt.bar(x + width/2, comparison_data['MLP'][:2], width, label='MLP', alpha=0.8)
    
    plt.xlabel('Métricas')
    plt.ylabel('Erro')
    plt.title('Comparação de Erros - Regressores')
    plt.xticks(x, ['RMSE', 'MAE'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: R² (maior é melhor)
    plt.subplot(2, 2, 2)
    plt.bar(['CART', 'MLP'], [comparison_data['CART'][2], comparison_data['MLP'][2]], alpha=0.8)
    plt.ylabel('R²')
    plt.title('R² - Regressores')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Validação cruzada
    plt.subplot(2, 2, 3)
    cv_data = [cart_results['cv_scores'], mlp_results['cv_scores']]
    plt.boxplot(cv_data, labels=['CART', 'MLP'])
    plt.ylabel('RMSE (CV)')
    plt.title('Validação Cruzada - Regressores')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Predições vs Real
    plt.subplot(2, 2, 4)
    plt.scatter(cart_results['y_test'], cart_results['y_pred_test'], alpha=0.6, label='CART')
    plt.scatter(mlp_results['y_test'], mlp_results['y_pred_test'], alpha=0.6, label='MLP')
    plt.plot([cart_results['y_test'].min(), cart_results['y_test'].max()], 
             [cart_results['y_test'].min(), cart_results['y_test'].max()], 'r--', alpha=0.8)
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.title('Predições vs Real')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regressor_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Relatório de comparação
    comparison_report_path = os.path.join(output_dir, 'regressor_comparison_report.txt')
    with open(comparison_report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE COMPARAÇÃO ENTRE REGRESSORES ===\n\n")
        
        f.write("1. MÉTRICAS DE PERFORMANCE:\n")
        f.write("-" * 50 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("2. VALIDAÇÃO CRUZADA:\n")
        f.write("-" * 50 + "\n")
        f.write(f"CART - Média RMSE: {cart_results['cv_mean']:.4f}, Std: {cart_results['cv_std']:.4f}\n")
        f.write(f"MLP - Média RMSE: {mlp_results['cv_mean']:.4f}, Std: {mlp_results['cv_std']:.4f}\n\n")
        
        f.write("3. ANÁLISE COMPARATIVA:\n")
        f.write("-" * 50 + "\n")
        
        # Determina o melhor algoritmo para cada métrica
        best_rmse = "CART" if cart_results['metrics']['rmse'] < mlp_results['metrics']['rmse'] else "MLP"
        best_mae = "CART" if cart_results['metrics']['mae'] < mlp_results['metrics']['mae'] else "MLP"
        best_r2 = "CART" if cart_results['metrics']['r2'] > mlp_results['metrics']['r2'] else "MLP"
        
        f.write(f"Melhor RMSE: {best_rmse}\n")
        f.write(f"Melhor MAE: {best_mae}\n")
        f.write(f"Melhor R²: {best_r2}\n\n")
        
        f.write("4. RECOMENDAÇÕES:\n")
        f.write("-" * 50 + "\n")
        f.write("• CART é mais interpretável e pode capturar relações não-lineares\n")
        f.write("• MLP é mais flexível mas pode ser menos interpretável\n")
        f.write("• Para este problema específico, recomenda-se usar ambos os modelos\n")
        f.write("  e escolher baseado no contexto de uso (interpretabilidade vs flexibilidade)\n")
    
    print(f"Relatório de comparação salvo em: {comparison_report_path}")
    print(f"Gráficos de comparação salvos em: {output_dir}")

def generate_prediction_report(df, cart_class_results, mlp_class_results, cart_reg_results, mlp_reg_results, output_dir):
    """
    Gera um relatório TXT com as labels preditas para cada vítima por ambos os algoritmos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Mapeia os códigos numéricos para nomes das classes (1-4)
    class_names = {
        1: 'CRÍTICO',
        2: 'INSTÁVEL', 
        3: 'POTENCIALMENTE ESTÁVEL',
        4: 'ESTÁVEL'
    }
    
    # Calcula o percentual de acerto geral para ambos classificadores
    cart_accuracy = accuracy_score(df['label'], cart_class_results['y_pred_all'])
    mlp_accuracy = accuracy_score(df['label'], mlp_class_results['y_pred_all'])
    
    # Gera o relatório
    report_path = os.path.join(output_dir, 'prediction_report_comparison.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE PREDIÇÕES - COMPARAÇÃO ENTRE ALGORITMOS ===\n\n")
        f.write("CLASSIFICADORES:\n")
        f.write(f"Acurácia CART: {cart_accuracy:.4f} ({cart_accuracy*100:.2f}%)\n")
        f.write(f"Acurácia MLP: {mlp_accuracy:.4f} ({mlp_accuracy*100:.2f}%)\n\n")
        
        f.write("REGRESSORES:\n")
        f.write(f"RMSE CART: {cart_reg_results['metrics']['rmse']:.4f}\n")
        f.write(f"RMSE MLP: {mlp_reg_results['metrics']['rmse']:.4f}\n")
        f.write(f"R² CART: {cart_reg_results['metrics']['r2']:.4f}\n")
        f.write(f"R² MLP: {mlp_reg_results['metrics']['r2']:.4f}\n\n")
        
        f.write("=" * 140 + "\n")
        f.write("ID | pSist | pDiast | qPA | Pulso | Resp | Grav Real | Label Real | CART Class | MLP Class | CART Grav | MLP Grav | CART Acerto | MLP Acerto\n")
        f.write("=" * 140 + "\n")
        
        cart_class_correct = 0
        mlp_class_correct = 0
        class_agreement = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            real_label = int(row['label'])
            real_grav = row['grav']
            cart_class_pred = int(cart_class_results['y_pred_all'][i])
            mlp_class_pred = int(mlp_class_results['y_pred_all'][i])
            cart_grav_pred = cart_reg_results['y_pred_all'][i]
            mlp_grav_pred = mlp_reg_results['y_pred_all'][i]
            
            cart_class_is_correct = real_label == cart_class_pred
            mlp_class_is_correct = real_label == mlp_class_pred
            class_models_agree = cart_class_pred == mlp_class_pred
            
            if cart_class_is_correct:
                cart_class_correct += 1
            if mlp_class_is_correct:
                mlp_class_correct += 1
            if class_models_agree:
                class_agreement += 1
            
            f.write(f"{int(row['id']):3d} | {row['pSist']:5.1f} | {row['pDiast']:6.1f} | {row['qPA']:3.1f} | {row['pulso']:5.1f} | {row['resp']:4.1f} | {real_grav:8.2f} | {class_names[real_label]:>20} | {class_names[cart_class_pred]:>9} | {class_names[mlp_class_pred]:>9} | {cart_grav_pred:8.2f} | {mlp_grav_pred:8.2f} | {'✓' if cart_class_is_correct else '✗':>11} | {'✓' if mlp_class_is_correct else '✗':>11}\n")
        
        f.write("=" * 140 + "\n")
        f.write(f"\nResumo:\n")
        f.write(f"- Total de vítimas: {len(df)}\n")
        f.write(f"- CART classificação correta: {cart_class_correct} ({cart_accuracy*100:.2f}%)\n")
        f.write(f"- MLP classificação correta: {mlp_class_correct} ({mlp_accuracy*100:.2f}%)\n")
        f.write(f"- Concordância entre classificadores: {class_agreement} ({class_agreement/len(df)*100:.2f}%)\n")
        
        # Análise por classe
        f.write(f"\nAnálise por Classe:\n")
        f.write("-" * 80 + "\n")
        for class_code, class_name in class_names.items():
            class_mask = df['label'] == class_code
            class_total = class_mask.sum()
            if class_total > 0:
                cart_class_correct_count = ((df['label'] == class_code) & (cart_class_results['y_pred_all'] == class_code)).sum()
                mlp_class_correct_count = ((df['label'] == class_code) & (mlp_class_results['y_pred_all'] == class_code)).sum()
                
                cart_class_acc = cart_class_correct_count / class_total
                mlp_class_acc = mlp_class_correct_count / class_total
                
                f.write(f"{class_name}:\n")
                f.write(f"  CART: {cart_class_correct_count}/{class_total} ({cart_class_acc*100:.1f}%)\n")
                f.write(f"  MLP: {mlp_class_correct_count}/{class_total} ({mlp_class_acc*100:.1f}%)\n")
    
    print(f"\nRelatório de predições salvo em: {report_path}")
    return report_path

def save_models(cart_class_results, mlp_class_results, cart_reg_results, mlp_reg_results, output_dir):
    """
    Salva os modelos treinados e os scalers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva os modelos classificadores
    cart_class_path = os.path.join(output_dir, 'cart_classifier.pkl')
    cart_class_scaler_path = os.path.join(output_dir, 'cart_classifier_scaler.pkl')
    
    with open(cart_class_path, 'wb') as f:
        pickle.dump(cart_class_results['classifier'], f)
    with open(cart_class_scaler_path, 'wb') as f:
        pickle.dump(cart_class_results['scaler'], f)
    
    mlp_class_path = os.path.join(output_dir, 'mlp_classifier.pkl')
    mlp_class_scaler_path = os.path.join(output_dir, 'mlp_classifier_scaler.pkl')
    
    with open(mlp_class_path, 'wb') as f:
        pickle.dump(mlp_class_results['classifier'], f)
    with open(mlp_class_scaler_path, 'wb') as f:
        pickle.dump(mlp_class_results['scaler'], f)
    
    # Salva os modelos regressores
    cart_reg_path = os.path.join(output_dir, 'cart_regressor.pkl')
    cart_reg_scaler_path = os.path.join(output_dir, 'cart_regressor_scaler.pkl')
    
    with open(cart_reg_path, 'wb') as f:
        pickle.dump(cart_reg_results['regressor'], f)
    with open(cart_reg_scaler_path, 'wb') as f:
        pickle.dump(cart_reg_results['scaler'], f)
    
    mlp_reg_path = os.path.join(output_dir, 'mlp_regressor.pkl')
    mlp_reg_scaler_path = os.path.join(output_dir, 'mlp_regressor_scaler.pkl')
    
    with open(mlp_reg_path, 'wb') as f:
        pickle.dump(mlp_reg_results['regressor'], f)
    with open(mlp_reg_scaler_path, 'wb') as f:
        pickle.dump(mlp_reg_results['scaler'], f)
    
    print(f"\nModelos salvos em:")
    print(f"  CART Classificador: {cart_class_path}")
    print(f"  CART Classificador Scaler: {cart_class_scaler_path}")
    print(f"  MLP Classificador: {mlp_class_path}")
    print(f"  MLP Classificador Scaler: {mlp_class_scaler_path}")
    print(f"  CART Regressor: {cart_reg_path}")
    print(f"  CART Regressor Scaler: {cart_reg_scaler_path}")
    print(f"  MLP Regressor: {mlp_reg_path}")
    print(f"  MLP Regressor Scaler: {mlp_reg_scaler_path}")

def train_all_models(X, y_class, y_reg, class_names):
    """
    Função principal que treina todos os modelos (classificadores e regressores) e retorna os resultados
    """
    print("\n" + "="*80)
    print("INICIANDO TREINAMENTO DE TODOS OS MODELOS")
    print("="*80)
    
    # Treina os classificadores
    print("\n" + "="*60)
    print("TREINANDO CLASSIFICADORES")
    print("="*60)
    cart_class_results, cart_class_all = train_cart_classifier_configs(X, y_class, class_names)
    mlp_class_results, mlp_class_all = train_mlp_classifier_configs(X, y_class, class_names)
    
    # Treina os regressores
    print("\n" + "="*60)
    print("TREINANDO REGRESSORES")
    print("="*60)
    cart_reg_results, cart_reg_all = train_cart_regressor_configs(X, y_reg)
    mlp_reg_results, mlp_reg_all = train_mlp_regressor_configs(X, y_reg)
    
    return cart_class_results, mlp_class_results, cart_reg_results, mlp_reg_results

def main():
    # Configurações
    dataset_path = "datasets/data_4000v/env_vital_signals.txt"
    output_dir = "mas/models"
    
    print("=== TREINAMENTO DOS CLASSIFICADORES E REGRESSORES DE VÍTIMAS ===")
    print(f"Dataset: {dataset_path}")
    print(f"Diretório de saída: {output_dir}")
    print(f"Algoritmos: CART (Decision Tree) e MLP (Neural Network)")
    print(f"Tarefas: Classificação (4 classes) e Regressão (valor de gravidade)")
    
    # Verifica se o arquivo existe
    if not os.path.exists(dataset_path):
        print(f"ERRO: Arquivo não encontrado: {dataset_path}")
        return
    
    # Carrega os dados
    df = load_vital_signals_data(dataset_path)
    
    # Prepara as features
    X, y_class, y_reg, features = prepare_features(df)
    class_names = ['CRÍTICO', 'INSTÁVEL', 'POTENCIALMENTE ESTÁVEL', 'ESTÁVEL']
    
    # Treina todos os modelos
    cart_class_results, mlp_class_results, cart_reg_results, mlp_reg_results = train_all_models(X, y_class, y_reg, class_names)
    
    # Gera análises de explicabilidade
    generate_explainability_analysis(cart_class_results, mlp_class_results, X, y_class, features, class_names, output_dir)
    
    # Compara os classificadores
    compare_classifiers(cart_class_results, mlp_class_results, output_dir)
    
    # Compara os regressores
    compare_regressors(cart_reg_results, mlp_reg_results, output_dir)
    
    # Gera o relatório de predições
    report_path = generate_prediction_report(df, cart_class_results, mlp_class_results, cart_reg_results, mlp_reg_results, output_dir)
    
    # Salva os modelos
    save_models(cart_class_results, mlp_class_results, cart_reg_results, mlp_reg_results, output_dir)
    
    print("\n" + "="*80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print("✓ Quatro modelos treinados (CART e MLP para classificação e regressão)")
    print("✓ 3 configurações testadas para cada modelo")
    print("✓ Validação cruzada executada para todos")
    print("✓ Análises de explicabilidade geradas")
    print("✓ Comparação entre algoritmos realizada")
    print("✓ Modelos salvos para uso pelos rescuers")
    print(f"✓ Relatórios detalhados disponíveis em: {output_dir}")
    print("\nOs classificadores e regressores estão prontos para ser usados pelos rescuers!")

if __name__ == "__main__":
    main() 