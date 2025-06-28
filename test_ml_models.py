#!/usr/bin/env python3
"""
Script para testar apenas regressão e classificação usando modelos já treinados.
Este script carrega modelos salvos e faz predições em novos dados ou nos dados de teste.

Uso:
    python test_ml_models.py --dataset datasets/data_800v/env_vital_signals.txt
    python test_ml_models.py --dataset datasets/data_4000v/env_vital_signals.txt --models-dir mas/models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

def load_vital_signals_data(file_path):
    """
    Carrega os dados de sinais vitais do arquivo, detectando automaticamente o delimitador
    """
    print(f"Carregando dados de: {file_path}")
    columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'grav', 'label']
    
    try:
        # Tenta ler por espaço/tabulação
        df = pd.read_csv(file_path, delim_whitespace=True, names=columns)
        if df.isnull().sum().sum() > 0:
            # Se muitos NaNs, tenta por vírgula
            df2 = pd.read_csv(file_path, sep=',', names=columns)
            # Se o segundo método tem menos NaNs, usa ele
            if df2.isnull().sum().sum() < df.isnull().sum().sum():
                df = df2
        print(f"Dados carregados: {len(df)} vítimas")
        print(f"Colunas: {list(df.columns)}")
        # Verifica e trata valores NaN
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"Valores NaN encontrados: {nan_counts.to_dict()}")
            print("Removendo linhas com valores NaN...")
            df_original = df.copy()
            df = df.dropna()
            print(f"Linhas removidas: {len(df_original) - len(df)}")
            print(f"Dados restantes: {len(df)} vítimas")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        print("Amostra do arquivo:")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                print(line.strip())
                if i > 4:
                    break
        return None

def prepare_features(df):
    """
    Prepara as features para predição (excluindo grav e label)
    """
    # Features: pSist, pDiast, qPA, pulso, resp
    features = ['pSist', 'pDiast', 'qPA', 'pulso', 'resp']
    X = df[features].values
    y_class = df['label'].values if 'label' in df.columns else None
    y_reg = df['grav'].values if 'grav' in df.columns else None
    
    return X, y_class, y_reg, features

def load_models(models_dir):
    """
    Carrega os modelos treinados do diretório especificado
    """
    print(f"Carregando modelos de: {models_dir}")
    
    models = {}
    
    # Lista de modelos para carregar
    model_files = [
        ('cart_classifier', 'cart_classifier.pkl'),
        ('cart_classifier_scaler', 'cart_classifier_scaler.pkl'),
        ('mlp_classifier', 'mlp_classifier.pkl'),
        ('mlp_classifier_scaler', 'mlp_classifier_scaler.pkl'),
        ('cart_regressor', 'cart_regressor.pkl'),
        ('cart_regressor_scaler', 'cart_regressor_scaler.pkl'),
        ('mlp_regressor', 'mlp_regressor.pkl'),
        ('mlp_regressor_scaler', 'mlp_regressor_scaler.pkl')
    ]
    
    for model_name, filename in model_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"✓ {model_name} carregado")
            except Exception as e:
                print(f"✗ Erro ao carregar {model_name}: {e}")
        else:
            print(f"✗ Arquivo não encontrado: {filepath}")
    
    return models

def predict_classification(X, models):
    """
    Faz predições de classificação usando os modelos carregados
    """
    predictions = {}
    
    # Predição com CART Classifier
    if 'cart_classifier' in models and 'cart_classifier_scaler' in models:
        X_scaled = models['cart_classifier_scaler'].transform(X)
        predictions['cart'] = models['cart_classifier'].predict(X_scaled)
        predictions['cart_proba'] = models['cart_classifier'].predict_proba(X_scaled)
    
    # Predição com MLP Classifier
    if 'mlp_classifier' in models and 'mlp_classifier_scaler' in models:
        X_scaled = models['mlp_classifier_scaler'].transform(X)
        predictions['mlp'] = models['mlp_classifier'].predict(X_scaled)
        predictions['mlp_proba'] = models['mlp_classifier'].predict_proba(X_scaled)
    
    return predictions

def predict_regression(X, models):
    """
    Faz predições de regressão usando os modelos carregados
    """
    predictions = {}
    
    # Predição com CART Regressor
    if 'cart_regressor' in models and 'cart_regressor_scaler' in models:
        X_scaled = models['cart_regressor_scaler'].transform(X)
        predictions['cart'] = models['cart_regressor'].predict(X_scaled)
    
    # Predição com MLP Regressor
    if 'mlp_regressor' in models and 'mlp_regressor_scaler' in models:
        X_scaled = models['mlp_regressor_scaler'].transform(X)
        predictions['mlp'] = models['mlp_regressor'].predict(X_scaled)
    
    return predictions

def evaluate_classification(y_true, predictions, class_names):
    """
    Avalia as predições de classificação
    """
    results = {}
    
    for model_name, y_pred in predictions.items():
        if model_name.endswith('_proba'):
            continue
            
        print(f"\n=== Avaliação {model_name.upper()} ===")
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Relatório detalhado
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
    
    return results

def evaluate_regression(y_true, predictions):
    """
    Avalia as predições de regressão
    """
    results = {}
    
    for model_name, y_pred in predictions.items():
        print(f"\n=== Avaliação {model_name.upper()} ===")
        
        # Métricas de regressão
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        results[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    return results

def plot_results(df, class_results, reg_results, output_dir):
    """
    Gera gráficos dos resultados
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Gráfico 1: Comparação de acurácia dos classificadores
    if class_results:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        model_names = list(class_results.keys())
        accuracies = [class_results[model]['accuracy'] for model in model_names]
        plt.bar(model_names, accuracies, alpha=0.8)
        plt.title('Acurácia dos Classificadores')
        plt.ylabel('Acurácia')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Gráfico 2: Matriz de confusão (primeiro classificador)
        if model_names:
            first_model = model_names[0]
            y_pred = class_results[first_model]['predictions']
            if 'label' in df.columns:
                cm = confusion_matrix(df['label'], y_pred)
                plt.subplot(2, 3, 2)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Matriz de Confusão - {first_model.upper()}')
                plt.ylabel('Real')
                plt.xlabel('Predito')
    
    # Gráfico 3: Comparação de RMSE dos regressores
    if reg_results:
        plt.subplot(2, 3, 3)
        model_names = list(reg_results.keys())
        rmse_values = [reg_results[model]['rmse'] for model in model_names]
        plt.bar(model_names, rmse_values, alpha=0.8)
        plt.title('RMSE dos Regressores')
        plt.ylabel('RMSE')
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        # Gráfico 4: Predições vs Real (primeiro regressor)
        if model_names and 'grav' in df.columns:
            first_model = model_names[0]
            y_pred = reg_results[first_model]['predictions']
            plt.subplot(2, 3, 4)
            plt.scatter(df['grav'], y_pred, alpha=0.6)
            plt.plot([df['grav'].min(), df['grav'].max()], 
                     [df['grav'].min(), df['grav'].max()], 'r--', alpha=0.8)
            plt.xlabel('Gravidade Real')
            plt.ylabel('Gravidade Predita')
            plt.title(f'Predições vs Real - {first_model.upper()}')
    
    # Gráfico 5: Distribuição das predições de classificação
    if class_results and 'label' in df.columns:
        plt.subplot(2, 3, 5)
        for model_name in class_results.keys():
            y_pred = class_results[model_name]['predictions']
            plt.hist(y_pred, alpha=0.7, label=model_name.upper(), bins=4)
        plt.xlabel('Classe Predita')
        plt.ylabel('Frequência')
        plt.title('Distribuição das Predições de Classificação')
        plt.legend()
    
    # Gráfico 6: Distribuição das predições de regressão
    if reg_results:
        plt.subplot(2, 3, 6)
        for model_name in reg_results.keys():
            y_pred = reg_results[model_name]['predictions']
            plt.hist(y_pred, alpha=0.7, label=model_name.upper(), bins=20)
        plt.xlabel('Gravidade Predita')
        plt.ylabel('Frequência')
        plt.title('Distribuição das Predições de Regressão')
        plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'test_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficos salvos em: {plot_path}")

def save_predictions(df, class_results, reg_results, output_dir):
    """
    Salva as predições em um arquivo CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cria um DataFrame com os resultados
    results_df = df.copy()
    
    # Adiciona predições de classificação
    for model_name, results in class_results.items():
        results_df[f'{model_name}_class_pred'] = results['predictions']
        if f'{model_name}_proba' in class_results:
            proba = class_results[f'{model_name}_proba']
            for i in range(proba.shape[1]):
                results_df[f'{model_name}_class_proba_{i}'] = proba[:, i]
    
    # Adiciona predições de regressão
    for model_name, results in reg_results.items():
        results_df[f'{model_name}_reg_pred'] = results['predictions']
    
    # Salva o arquivo
    output_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predições salvas em: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Testa modelos de ML para classificação e regressão")
    parser.add_argument("--dataset", required=True, help="Caminho para o arquivo de dados")
    parser.add_argument("--models-dir", default="mas/models", help="Diretório com os modelos treinados")
    parser.add_argument("--output-dir", default="test_results", help="Diretório para salvar resultados")
    
    args = parser.parse_args()
    
    print("=== TESTE DE MODELOS DE MACHINE LEARNING ===")
    print(f"Dataset: {args.dataset}")
    print(f"Modelos: {args.models_dir}")
    print(f"Saída: {args.output_dir}")
    
    # Verifica se o arquivo de dados existe
    if not os.path.exists(args.dataset):
        print(f"ERRO: Arquivo não encontrado: {args.dataset}")
        return
    
    # Verifica se o diretório de modelos existe
    if not os.path.exists(args.models_dir):
        print(f"ERRO: Diretório de modelos não encontrado: {args.models_dir}")
        return
    
    # Carrega os dados
    df = load_vital_signals_data(args.dataset)
    if df is None:
        return
    
    # Prepara as features
    X, y_class, y_reg, features = prepare_features(df)
    print(f"Features: {features}")
    print(f"Shape dos dados: {X.shape}")
    
    # Carrega os modelos
    models = load_models(args.models_dir)
    if not models:
        print("ERRO: Nenhum modelo foi carregado")
        return
    
    # Faz predições
    print("\n=== FAZENDO PREDIÇÕES ===")
    
    class_results = {}
    reg_results = {}
    
    # Predições de classificação
    if any('classifier' in model for model in models.keys()):
        print("Fazendo predições de classificação...")
        class_predictions = predict_classification(X, models)
        
        if y_class is not None:
            class_names = ['CRÍTICO', 'INSTÁVEL', 'POTENCIALMENTE ESTÁVEL', 'ESTÁVEL']
            class_results = evaluate_classification(y_class, class_predictions, class_names)
        else:
            print("Aviso: Dados de classificação não disponíveis para avaliação")
            class_results = {k: {'predictions': v} for k, v in class_predictions.items() if not k.endswith('_proba')}
    
    # Predições de regressão
    if any('regressor' in model for model in models.keys()):
        print("Fazendo predições de regressão...")
        reg_predictions = predict_regression(X, models)
        
        if y_reg is not None:
            reg_results = evaluate_regression(y_reg, reg_predictions)
        else:
            print("Aviso: Dados de regressão não disponíveis para avaliação")
            reg_results = {k: {'predictions': v} for k, v in reg_predictions.items()}
    
    # Gera gráficos
    if class_results or reg_results:
        print("\n=== GERANDO GRÁFICOS ===")
        plot_results(df, class_results, reg_results, args.output_dir)
        
        # Salva predições
        print("\n=== SALVANDO PREDIÇÕES ===")
        save_predictions(df, class_results, reg_results, args.output_dir)
    
    print("\n=== TESTE CONCLUÍDO ===")

if __name__ == "__main__":
    main() 