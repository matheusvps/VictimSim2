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

def train_classifier(X, y, class_names):
    """
    Treina o classificador Random Forest e retorna predições para todos os dados
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
    
    # Avalia o modelo no conjunto de teste
    y_pred_test = classifier.predict(X_test_scaled)
    
    print("\n=== Resultados da Avaliação (Conjunto de Teste) ===")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_test):.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # Faz predições para todos os dados
    X_scaled = scaler.transform(X)
    y_pred_all = classifier.predict(X_scaled)
    
    return classifier, scaler, y_pred_all

def generate_prediction_report(df, y_pred, output_dir):
    """
    Gera um relatório TXT com as labels preditas para cada vítima e percentual de acerto
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Mapeia os códigos numéricos para nomes das classes (1-4)
    class_names = {
        1: 'CRÍTICO',
        2: 'INSTÁVEL', 
        3: 'POTENCIALMENTE ESTÁVEL',
        4: 'ESTÁVEL'
    }
    
    # Calcula o percentual de acerto geral
    accuracy = accuracy_score(df['label'], y_pred)
    
    # Gera o relatório
    report_path = os.path.join(output_dir, 'prediction_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE PREDIÇÕES DO CLASSIFICADOR DE VÍTIMAS ===\n\n")
        f.write(f"Percentual de Acerto Geral: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("=" * 80 + "\n")
        f.write("ID | pSist | pDiast | qPA | Pulso | Resp | Grav | Label Real | Label Predita | Acerto\n")
        f.write("=" * 80 + "\n")
        
        correct_predictions = 0
        for i, (idx, row) in enumerate(df.iterrows()):
            real_label = int(row['label'])
            pred_label = int(y_pred[i])
            is_correct = real_label == pred_label
            
            if is_correct:
                correct_predictions += 1
            
            f.write(f"{int(row['id']):3d} | {row['pSist']:5.1f} | {row['pDiast']:6.1f} | {row['qPA']:3.1f} | {row['pulso']:5.1f} | {row['resp']:4.1f} | {row['grav']:4.1f} | {class_names[real_label]:>20} | {class_names[pred_label]:>18} | {'✓' if is_correct else '✗'}\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"\nResumo:\n")
        f.write(f"- Total de vítimas: {len(df)}\n")
        f.write(f"- Predições corretas: {correct_predictions}\n")
        f.write(f"- Predições incorretas: {len(df) - correct_predictions}\n")
        f.write(f"- Percentual de acerto: {accuracy*100:.2f}%\n\n")
        
        # Análise por classe
        f.write("Análise por Classe:\n")
        f.write("-" * 50 + "\n")
        for class_code, class_name in class_names.items():
            class_mask = df['label'] == class_code
            class_total = class_mask.sum()
            if class_total > 0:
                class_correct = ((df['label'] == class_code) & (y_pred == class_code)).sum()
                class_accuracy = class_correct / class_total
                f.write(f"{class_name}: {class_correct}/{class_total} ({class_accuracy*100:.1f}%)\n")
    
    print(f"\nRelatório de predições salvo em: {report_path}")
    return report_path

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
    
    # Treina o classificador e obtém predições
    classifier, scaler, y_pred = train_classifier(X, y, ['CRÍTICO', 'INSTÁVEL', 'POTENCIALMENTE ESTÁVEL', 'ESTÁVEL'])
    
    # Gera o relatório de predições
    report_path = generate_prediction_report(df, y_pred, output_dir)
    
    # Salva o modelo
    save_model(classifier, scaler, output_dir)
    
    print("\n=== Treinamento Concluído ===")
    print("O classificador está pronto para ser usado pelos rescuers!")
    print(f"Relatório detalhado disponível em: {report_path}")

if __name__ == "__main__":
    main() 