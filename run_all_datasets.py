#!/usr/bin/env python3
import os
import subprocess
import time
from datetime import datetime
import sys

def run_simulation(dataset_path, config_path):
    print(f"\n{'='*80}")
    print(f"Iniciando simulação para dataset: {dataset_path}")
    print(f"{'='*80}\n")
    
    # Executa o main.py com o dataset atual
    cmd = ["python", os.path.join("mas", "main.py"), dataset_path, config_path]
    
    # Adiciona os diretórios necessários ao PYTHONPATH
    env = os.environ.copy()
    current_dir = os.path.abspath(os.getcwd())
    env["PYTHONPATH"] = os.pathsep.join([current_dir, os.path.join(current_dir, "mas")])
    
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True, text=True, env=env)
    end_time = time.time()
    
    # Cria diretório para logs se não existir
    os.makedirs("simulation_logs", exist_ok=True)
    
    # Gera nome do arquivo de log baseado no dataset
    dataset_name = os.path.basename(dataset_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("simulation_logs", f"{dataset_name}_{timestamp}.log")
    
    # Salva o log
    with open(log_file, "w") as f:
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Tempo de execução: {end_time - start_time:.2f} segundos\n")
        f.write("\n=== Saída ===\n")
        f.write(process.stdout)
        f.write("\n=== Erros ===\n")
        f.write(process.stderr)
    
    print(f"\nSimulação concluída em {end_time - start_time:.2f} segundos")
    print(f"Log salvo em: {log_file}")
    
    if process.returncode != 0:
        print(f"\nERRO: A simulação falhou com código {process.returncode}")
        print("Verifique o arquivo de log para mais detalhes.")

def get_dataset_choice(datasets):
    while True:
        print("\nEscolha o dataset para simulação:")
        print("0. Executar todos os datasets")
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset}")
        
        try:
            choice = int(input("\nDigite o número do dataset (0 para todos): "))
            if 0 <= choice <= len(datasets):
                return choice
            print("Opção inválida! Por favor, escolha um número válido.")
        except ValueError:
            print("Por favor, digite um número válido.")

def main():
    # Diretório base dos datasets
    datasets_dir = "datasets"
    
    # Configuração padrão
    config_path = "cfg_1"
    
    # Lista todos os datasets
    datasets = [d for d in os.listdir(datasets_dir) 
               if os.path.isdir(os.path.join(datasets_dir, d)) 
               and d.startswith("data_")]
    
    # Ordena os datasets por tamanho (do menor para o maior)
    datasets.sort(key=lambda x: int(x.split('_')[1].replace('v', '')))
    
    # Obtém a escolha do usuário
    choice = get_dataset_choice(datasets)
    
    print("\nIniciando simulações...")
    
    if choice == 0:
        # Executa todos os datasets
        for dataset in datasets:
            dataset_path = os.path.join(datasets_dir, dataset)
            run_simulation(dataset_path, config_path)
            time.sleep(1)
    else:
        # Executa apenas o dataset escolhido
        dataset = datasets[choice - 1]
        dataset_path = os.path.join(datasets_dir, dataset)
        run_simulation(dataset_path, config_path)

if __name__ == "__main__":
    main() 