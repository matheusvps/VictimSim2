import sys
import os
import time

# Adiciona o diretório atual ao path para encontrar os módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

## importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer
from vs.constants import VS

def main(data_folder_name, config_ag_folder_name):
   
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # Instantiate the environment
    env = Env(data_folder)
    
    # Instantiate master_rescuer
    # This agent unifies the maps and instantiate other 3 agents
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)   # 4 is the number of explorer agents

    # Explorer needs to know rescuer to send the map 
    # that's why rescuer is instatiated before
    # Direções iniciais para cada explorador:
    # 0: direita
    # 2: baixo
    # 4: esquerda
    # 6: cima
    directions = [0, 2, 4, 6]
    
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer, directions[exp-1])

    # Run the environment simulator
    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    # Valores padrão
    data_folder_name = os.path.join("datasets", "data_300v_90x90")
    config_ag_folder_name = "cfg_1"
    
    # Se houver argumentos na linha de comando, use-os
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    if len(sys.argv) > 2:
        config_ag_folder_name = sys.argv[2]
        
    main(data_folder_name, config_ag_folder_name)
