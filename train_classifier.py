#!/usr/bin/env python3
"""
Script para treinar classificadores usando o dataset 4000v
e salvar os modelos treinados para uso posterior pelos rescuers.

Implementa dois algoritmos diferentes:
1. CART (Decision Tree) - Árvore de Decisão
2. Perceptron (Neural Network) - Rede Neural Perceptron

Inclui validação cruzada, explicabilidade (LIME/SHAP) e comparação entre algoritmos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
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
    y = df['label'].values
    
    return X, y, features

def train_cart_classifier(X, y, class_names, cv_folds=5):
    """
    Treina um classificador CART (Decision Tree) com validação cruzada
    """
    print("\n" + "="*60)
    print("TREINANDO CLASSIFICADOR CART (DECISION TREE)")
    print("="*60)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configura o classificador CART
    cart_classifier = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        criterion='gini'
    )
    
    # Validação cruzada
    print(f"\nExecutando validação cruzada ({cv_folds} folds)...")
    cv_scores = cross_val_score(
        cart_classifier, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    print(f"Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Scores individuais: {cv_scores}")
    
    # Treina o modelo final
    print("\nTreinando modelo final...")
    cart_classifier.fit(X_train_scaled, y_train)
    
    # Avalia o modelo no conjunto de teste
    y_pred_test = cart_classifier.predict(X_test_scaled)
    y_pred_proba = cart_classifier.predict_proba(X_test_scaled)
    
    # Métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\n=== RESULTADOS CART (Conjunto de Teste) ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nMatriz de Confusão:")
    print(cm)
    
    # Relatório detalhado
    print(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # Faz predições para todos os dados
    X_scaled = scaler.transform(X)
    y_pred_all = cart_classifier.predict(X_scaled)
    
    return {
        'classifier': cart_classifier,
        'scaler': scaler,
        'y_pred_all': y_pred_all,
        'y_pred_test': y_pred_test,
        'y_test': y_test,
        'X_test_scaled': X_test_scaled,
        'cv_scores': cv_scores,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }

def train_perceptron_classifier(X, y, class_names, cv_folds=5):
    """
    Treina um classificador Perceptron (Neural Network) com validação cruzada
    """
    print("\n" + "="*60)
    print("TREINANDO CLASSIFICADOR PERCEPTRON (NEURAL NETWORK)")
    print("="*60)
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configura o classificador Perceptron
    perceptron_classifier = Perceptron(
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        eta0=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    # Validação cruzada
    print(f"\nExecutando validação cruzada ({cv_folds} folds)...")
    cv_scores = cross_val_score(
        perceptron_classifier, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    print(f"Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Scores individuais: {cv_scores}")
    
    # Treina o modelo final
    print("\nTreinando modelo final...")
    perceptron_classifier.fit(X_train_scaled, y_train)
    
    # Avalia o modelo no conjunto de teste
    y_pred_test = perceptron_classifier.predict(X_test_scaled)
    
    # Métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    print(f"\n=== RESULTADOS PERCEPTRON (Conjunto de Teste) ===")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\nMatriz de Confusão:")
    print(cm)
    
    # Relatório detalhado
    print(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))
    
    # Faz predições para todos os dados
    X_scaled = scaler.transform(X)
    y_pred_all = perceptron_classifier.predict(X_scaled)
    
    return {
        'classifier': perceptron_classifier,
        'scaler': scaler,
        'y_pred_all': y_pred_all,
        'y_pred_test': y_pred_test,
        'y_test': y_test,
        'X_test_scaled': X_test_scaled,
        'cv_scores': cv_scores,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }

def generate_explainability_analysis(cart_results, perceptron_results, X, y, features, class_names, output_dir):
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

def compare_algorithms(cart_results, perceptron_results, output_dir):
    """
    Compara os dois algoritmos e gera relatórios comparativos
    """
    print("\n" + "="*60)
    print("COMPARANDO ALGORITMOS")
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
        'Perceptron': [
            perceptron_results['metrics']['accuracy'],
            perceptron_results['metrics']['precision'],
            perceptron_results['metrics']['recall'],
            perceptron_results['metrics']['f1']
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
    plt.bar(x + width/2, comparison_data['Perceptron'], width, label='Perceptron', alpha=0.8)
    
    plt.xlabel('Métricas')
    plt.ylabel('Score')
    plt.title('Comparação de Métricas')
    plt.xticks(x, comparison_data['Métrica'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Validação cruzada
    plt.subplot(2, 2, 2)
    cv_data = [cart_results['cv_scores'], perceptron_results['cv_scores']]
    plt.boxplot(cv_data, labels=['CART', 'Perceptron'])
    plt.ylabel('Acurácia (CV)')
    plt.title('Validação Cruzada')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Matriz de confusão CART
    plt.subplot(2, 2, 3)
    cm_cart = confusion_matrix(cart_results['y_test'], cart_results['y_pred_test'])
    sns.heatmap(cm_cart, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - CART')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    # Subplot 4: Matriz de confusão Perceptron
    plt.subplot(2, 2, 4)
    cm_perceptron = confusion_matrix(perceptron_results['y_test'], perceptron_results['y_pred_test'])
    sns.heatmap(cm_perceptron, annot=True, fmt='d', cmap='Reds')
    plt.title('Matriz de Confusão - Perceptron')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Relatório de comparação
    comparison_report_path = os.path.join(output_dir, 'algorithm_comparison_report.txt')
    with open(comparison_report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE COMPARAÇÃO ENTRE ALGORITMOS ===\n\n")
        
        f.write("1. MÉTRICAS DE PERFORMANCE:\n")
        f.write("-" * 50 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("2. VALIDAÇÃO CRUZADA:\n")
        f.write("-" * 50 + "\n")
        f.write(f"CART - Média: {cart_results['cv_scores'].mean():.4f}, Std: {cart_results['cv_scores'].std():.4f}\n")
        f.write(f"Perceptron - Média: {perceptron_results['cv_scores'].mean():.4f}, Std: {perceptron_results['cv_scores'].std():.4f}\n\n")
        
        f.write("3. ANÁLISE COMPARATIVA:\n")
        f.write("-" * 50 + "\n")
        
        # Determina o melhor algoritmo para cada métrica
        best_accuracy = "CART" if cart_results['metrics']['accuracy'] > perceptron_results['metrics']['accuracy'] else "Perceptron"
        best_precision = "CART" if cart_results['metrics']['precision'] > perceptron_results['metrics']['precision'] else "Perceptron"
        best_recall = "CART" if cart_results['metrics']['recall'] > perceptron_results['metrics']['recall'] else "Perceptron"
        best_f1 = "CART" if cart_results['metrics']['f1'] > perceptron_results['metrics']['f1'] else "Perceptron"
        
        f.write(f"Melhor Acurácia: {best_accuracy}\n")
        f.write(f"Melhor Precisão: {best_precision}\n")
        f.write(f"Melhor Recall: {best_recall}\n")
        f.write(f"Melhor F1-Score: {best_f1}\n\n")
        
        f.write("4. RECOMENDAÇÕES:\n")
        f.write("-" * 50 + "\n")
        f.write("• CART é mais interpretável e pode capturar relações não-lineares\n")
        f.write("• Perceptron é mais rápido para predições mas pode ser menos interpretável\n")
        f.write("• Para este problema específico, recomenda-se usar ambos os modelos\n")
        f.write("  e escolher baseado no contexto de uso (interpretabilidade vs velocidade)\n")
    
    print(f"Relatório de comparação salvo em: {comparison_report_path}")
    print(f"Gráficos de comparação salvos em: {output_dir}")

def generate_prediction_report(df, cart_results, perceptron_results, output_dir):
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
    
    # Calcula o percentual de acerto geral para ambos
    cart_accuracy = accuracy_score(df['label'], cart_results['y_pred_all'])
    perceptron_accuracy = accuracy_score(df['label'], perceptron_results['y_pred_all'])
    
    # Gera o relatório
    report_path = os.path.join(output_dir, 'prediction_report_comparison.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== RELATÓRIO DE PREDIÇÕES - COMPARAÇÃO ENTRE ALGORITMOS ===\n\n")
        f.write(f"Acurácia CART: {cart_accuracy:.4f} ({cart_accuracy*100:.2f}%)\n")
        f.write(f"Acurácia Perceptron: {perceptron_accuracy:.4f} ({perceptron_accuracy*100:.2f}%)\n\n")
        f.write("=" * 120 + "\n")
        f.write("ID | pSist | pDiast | qPA | Pulso | Resp | Grav | Label Real | CART Pred | Perceptron Pred | CART Acerto | Perceptron Acerto\n")
        f.write("=" * 120 + "\n")
        
        cart_correct = 0
        perceptron_correct = 0
        agreement = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            real_label = int(row['label'])
            cart_pred = int(cart_results['y_pred_all'][i])
            perceptron_pred = int(perceptron_results['y_pred_all'][i])
            
            cart_is_correct = real_label == cart_pred
            perceptron_is_correct = real_label == perceptron_pred
            models_agree = cart_pred == perceptron_pred
            
            if cart_is_correct:
                cart_correct += 1
            if perceptron_is_correct:
                perceptron_correct += 1
            if models_agree:
                agreement += 1
            
            f.write(f"{int(row['id']):3d} | {row['pSist']:5.1f} | {row['pDiast']:6.1f} | {row['qPA']:3.1f} | {row['pulso']:5.1f} | {row['resp']:4.1f} | {row['grav']:4.1f} | {class_names[real_label]:>20} | {class_names[cart_pred]:>9} | {class_names[perceptron_pred]:>13} | {'✓' if cart_is_correct else '✗':>11} | {'✓' if perceptron_is_correct else '✗':>15}\n")
        
        f.write("=" * 120 + "\n")
        f.write(f"\nResumo:\n")
        f.write(f"- Total de vítimas: {len(df)}\n")
        f.write(f"- CART correto: {cart_correct} ({cart_accuracy*100:.2f}%)\n")
        f.write(f"- Perceptron correto: {perceptron_correct} ({perceptron_accuracy*100:.2f}%)\n")
        f.write(f"- Concordância entre modelos: {agreement} ({agreement/len(df)*100:.2f}%)\n")
        
        # Análise por classe
        f.write(f"\nAnálise por Classe:\n")
        f.write("-" * 80 + "\n")
        for class_code, class_name in class_names.items():
            class_mask = df['label'] == class_code
            class_total = class_mask.sum()
            if class_total > 0:
                cart_class_correct = ((df['label'] == class_code) & (cart_results['y_pred_all'] == class_code)).sum()
                perceptron_class_correct = ((df['label'] == class_code) & (perceptron_results['y_pred_all'] == class_code)).sum()
                
                cart_class_acc = cart_class_correct / class_total
                perceptron_class_acc = perceptron_class_correct / class_total
                
                f.write(f"{class_name}:\n")
                f.write(f"  CART: {cart_class_correct}/{class_total} ({cart_class_acc*100:.1f}%)\n")
                f.write(f"  Perceptron: {perceptron_class_correct}/{class_total} ({perceptron_class_acc*100:.1f}%)\n")
    
    print(f"\nRelatório de predições salvo em: {report_path}")
    return report_path

def save_models(cart_results, perceptron_results, output_dir):
    """
    Salva os modelos treinados e os scalers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o modelo CART
    cart_path = os.path.join(output_dir, 'cart_classifier.pkl')
    cart_scaler_path = os.path.join(output_dir, 'cart_scaler.pkl')
    
    with open(cart_path, 'wb') as f:
        pickle.dump(cart_results['classifier'], f)
    with open(cart_scaler_path, 'wb') as f:
        pickle.dump(cart_results['scaler'], f)
    
    # Salva o modelo Perceptron
    perceptron_path = os.path.join(output_dir, 'perceptron_classifier.pkl')
    perceptron_scaler_path = os.path.join(output_dir, 'perceptron_scaler.pkl')
    
    with open(perceptron_path, 'wb') as f:
        pickle.dump(perceptron_results['classifier'], f)
    with open(perceptron_scaler_path, 'wb') as f:
        pickle.dump(perceptron_results['scaler'], f)
    
    print(f"\nModelos salvos em:")
    print(f"  CART: {cart_path}")
    print(f"  CART Scaler: {cart_scaler_path}")
    print(f"  Perceptron: {perceptron_path}")
    print(f"  Perceptron Scaler: {perceptron_scaler_path}")

def train_classifier(X, y, class_names):
    """
    Função principal que treina ambos os classificadores e retorna os resultados
    """
    print("\n" + "="*80)
    print("INICIANDO TREINAMENTO DOS CLASSIFICADORES")
    print("="*80)
    
    # Treina o classificador CART
    cart_results = train_cart_classifier(X, y, class_names)
    
    # Treina o classificador Perceptron
    perceptron_results = train_perceptron_classifier(X, y, class_names)
    
    return cart_results, perceptron_results

def main():
    # Configurações
    dataset_path = "datasets/data_4000v/env_vital_signals.txt"
    output_dir = "mas/models"
    
    print("=== TREINAMENTO DOS CLASSIFICADORES DE VÍTIMAS ===")
    print(f"Dataset: {dataset_path}")
    print(f"Diretório de saída: {output_dir}")
    print(f"Algoritmos: CART (Decision Tree) e Perceptron (Neural Network)")
    
    # Verifica se o arquivo existe
    if not os.path.exists(dataset_path):
        print(f"ERRO: Arquivo não encontrado: {dataset_path}")
        return
    
    # Carrega os dados
    df = load_vital_signals_data(dataset_path)
    
    # Prepara as features
    X, y, features = prepare_features(df)
    class_names = ['CRÍTICO', 'INSTÁVEL', 'POTENCIALMENTE ESTÁVEL', 'ESTÁVEL']
    
    # Treina os classificadores
    cart_results, perceptron_results = train_classifier(X, y, class_names)
    
    # Gera análises de explicabilidade
    generate_explainability_analysis(cart_results, perceptron_results, X, y, features, class_names, output_dir)
    
    # Compara os algoritmos
    compare_algorithms(cart_results, perceptron_results, output_dir)
    
    # Gera o relatório de predições
    report_path = generate_prediction_report(df, cart_results, perceptron_results, output_dir)
    
    # Salva os modelos
    save_models(cart_results, perceptron_results, output_dir)
    
    print("\n" + "="*80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print("✓ Dois classificadores treinados (CART e Perceptron)")
    print("✓ Validação cruzada executada para ambos")
    print("✓ Análises de explicabilidade geradas")
    print("✓ Comparação entre algoritmos realizada")
    print("✓ Modelos salvos para uso pelos rescuers")
    print(f"✓ Relatórios detalhados disponíveis em: {output_dir}")
    print("\nOs classificadores estão prontos para ser usados pelos rescuers!")

if __name__ == "__main__":
    main() 