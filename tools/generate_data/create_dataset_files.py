#!/usr/bin/env python3
"""
Script para criar os arquivos necessários para datasets que só possuem env_vital_signals.txt
Gera: env_config.txt, env_obst.txt, env_victims.txt
"""

import os
import random
import argparse

def create_env_config(data_folder, grid_width=30, grid_height=30, window_width=800, window_height=800):
    """Cria o arquivo env_config.txt"""
    config_file = os.path.join(data_folder, "env_config.txt")
    
    config_content = f"""BASE 0,0
GRID_WIDTH {grid_width}
GRID_HEIGHT {grid_height}
WINDOW_WIDTH {window_width}
WINDOW_HEIGHT {window_height}
DELAY 0.0
STATS_PER_AG 1
STATS_ALL_AG 1"""
    
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print(f"Arquivo {config_file} criado")

def create_env_obst(data_folder):
    """Cria o arquivo env_obst.txt (vazio por padrão)"""
    obst_file = os.path.join(data_folder, "env_obst.txt")
    
    # Cria arquivo vazio (sem obstáculos)
    with open(obst_file, "w") as f:
        pass
    
    print(f"Arquivo {obst_file} criado (vazio)")

def create_env_victims(data_folder, grid_width=30, grid_height=30):
    """Cria o arquivo env_victims.txt com posições aleatórias"""
    victims_file = os.path.join(data_folder, "env_victims.txt")
    vital_signals_file = os.path.join(data_folder, "env_vital_signals.txt")
    
    # Conta o número de vítimas no arquivo de sinais vitais
    num_victims = 0
    if os.path.exists(vital_signals_file):
        with open(vital_signals_file, "r") as f:
            num_victims = sum(1 for line in f)
    
    if num_victims == 0:
        print("Erro: Não foi possível determinar o número de vítimas")
        return
    
    print(f"Gerando {num_victims} posições de vítimas...")
    
    # Gera posições aleatórias para as vítimas
    positions = []
    while len(positions) < num_victims:
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        
        # Evita posição (0,0) que é a base
        if (x, y) != (0, 0) and (x, y) not in positions:
            positions.append((x, y))
    
    # Ordena as posições
    positions.sort(key=lambda pos: (pos[0], pos[1]))
    
    # Escreve no arquivo
    with open(victims_file, "w") as f:
        for x, y in positions:
            f.write(f"{x},{y}\n")
    
    print(f"Arquivo {victims_file} criado com {num_victims} vítimas")

def main():
    parser = argparse.ArgumentParser(description="Cria arquivos necessários para datasets")
    parser.add_argument("data_folder", help="Pasta do dataset")
    parser.add_argument("--grid-width", type=int, default=30, help="Largura do grid")
    parser.add_argument("--grid-height", type=int, default=30, help="Altura do grid")
    parser.add_argument("--window-width", type=int, default=800, help="Largura da janela")
    parser.add_argument("--window-height", type=int, default=800, help="Altura da janela")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_folder):
        print(f"Erro: Pasta {args.data_folder} não existe")
        return
    
    print(f"Criando arquivos para dataset: {args.data_folder}")
    
    # Cria os arquivos
    create_env_config(args.data_folder, args.grid_width, args.grid_height, 
                     args.window_width, args.window_height)
    create_env_obst(args.data_folder)
    create_env_victims(args.data_folder, args.grid_width, args.grid_height)
    
    print("Arquivos criados com sucesso!")

if __name__ == "__main__":
    main() 