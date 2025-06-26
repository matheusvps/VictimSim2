#!/usr/bin/env python3
"""
Script para treinar um classificador usando o dataset 4000v
e salvar o modelo treinado para uso posterior pelos rescuers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pickle

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
    y = df['label'].values
    
    return X, y

def train_classifier(X, y):
    """
    Treina o classificador Random Forest
    """
    print("\nTreinando classificador Random Forest...")
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treina o classificador
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train_scaled, y_train)
    
    # Avalia o modelo
    y_pred = classifier.predict(X_test_scaled)
    
    print("\n=== Resultados da Avaliação ===")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['CRÍTICO', 'INSTÁVEL', 'POTENCIALMENTE ESTÁVEL', 'ESTÁVEL']))
    
    return classifier, scaler

def save_model(classifier, scaler, output_dir):
    """
    Salva o modelo treinado e o scaler
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o classificador
    classifier_path = os.path.join(output_dir, 'victim_classifier.pkl')
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    # Salva o scaler
    scaler_path = os.path.join(output_dir, 'victim_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModelo salvo em: {classifier_path}")
    print(f"Scaler salvo em: {scaler_path}")

def main():
    # Configurações
    dataset_path = "datasets/data_4000v/env_vital_signals.txt"
    output_dir = "mas/models"
    
    print("=== Treinamento do Classificador de Vítimas ===")
    print(f"Dataset: {dataset_path}")
    print(f"Diretório de saída: {output_dir}")
    
    # Verifica se o arquivo existe
    if not os.path.exists(dataset_path):
        print(f"ERRO: Arquivo não encontrado: {dataset_path}")
        return
    
    # Carrega os dados
    df = load_vital_signals_data(dataset_path)
    
    # Prepara as features
    X, y = prepare_features(df)
    
    # Treina o classificador
    classifier, scaler = train_classifier(X, y)
    
    # Salva o modelo
    save_model(classifier, scaler, output_dir)
    
    print("\n=== Treinamento Concluído ===")
    print("O classificador está pronto para ser usado pelos rescuers!")

if __name__ == "__main__":
    main() 