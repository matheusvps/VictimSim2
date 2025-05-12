# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
import time
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    DELAY = 0.005                   # delay em segundos entre ações
    
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        
        # Obtém a posição inicial do explorador
        self.base_x = self.get_env().dic["BASE"][0]  # posição x inicial
        self.base_y = self.get_env().dic["BASE"][1]  # posição y inicial
        self.x = self.base_x       # current x position relative to the base
        self.y = self.base_y       # current y position relative to the base
        
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.visited = set()       # conjunto de posições já visitadas
        self.visited.add((self.x, self.y))   # adiciona a posição inicial (base)
        
        # Identifica o explorador baseado no nome
        self.explorer_id = int(self.NAME[-1]) - 1  # EXPL_1 -> 0, EXPL_2 -> 1, etc.
        
        # Define a direção inicial e o padrão de exploração para cada explorador
        self.exploration_patterns = {
            0: {  # Explorador 1: Explora no sentido horário a partir do norte
                'initial_direction': 3,  # Começa indo para cima
                'direction_sequence': [3, 0, 1, 2],  # Norte -> Leste -> Sul -> Oeste
                'quadrant': {'x': (0.4, 0.6), 'y': (0, 0.4)}  # Área mais próxima do norte
            },
            1: {  # Explorador 2: Explora no sentido anti-horário a partir do leste
                'initial_direction': 0,  # Começa indo para direita
                'direction_sequence': [0, 3, 2, 1],  # Leste -> Norte -> Oeste -> Sul
                'quadrant': {'x': (0.6, 1), 'y': (0.4, 0.6)}  # Área mais próxima do leste
            },
            2: {  # Explorador 3: Explora no sentido horário a partir do sul
                'initial_direction': 1,  # Começa indo para baixo
                'direction_sequence': [1, 2, 3, 0],  # Sul -> Oeste -> Norte -> Leste
                'quadrant': {'x': (0.4, 0.6), 'y': (0.6, 1)}  # Área mais próxima do sul
            },
            3: {  # Explorador 4: Explora no sentido anti-horário a partir do oeste
                'initial_direction': 2,  # Começa indo para esquerda
                'direction_sequence': [2, 1, 0, 3],  # Oeste -> Sul -> Leste -> Norte
                'quadrant': {'x': (0, 0.4), 'y': (0.4, 0.6)}  # Área mais próxima do oeste
            }
        }
        
        # Inicializa as variáveis de exploração
        pattern = self.exploration_patterns[self.explorer_id]
        self.spiral_direction = pattern['initial_direction']
        self.direction_sequence = pattern['direction_sequence']
        self.quadrant_limits = pattern['quadrant']
        self.spiral_step = 1
        self.spiral_count = 0
        self.spiral_turns = 0

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        """ Gets the next position that can be explored (no wall and inside the grid)
            Uses a customized spiral pattern for each explorer
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        
        # Get the current map dimensions from the environment
        map_width = self.get_env().dic["GRID_WIDTH"]
        map_height = self.get_env().dic["GRID_HEIGHT"]
        
        # Calculate the absolute limits of the quadrant
        x_min = int(map_width * self.quadrant_limits['x'][0])
        x_max = int(map_width * self.quadrant_limits['x'][1])
        y_min = int(map_height * self.quadrant_limits['y'][0])
        y_max = int(map_height * self.quadrant_limits['y'][1])
        
        # Direções da espiral (direita, baixo, esquerda, cima)
        spiral_directions = [
            (1, 0),   # direita
            (0, 1),   # baixo
            (-1, 0),  # esquerda
            (0, -1)   # cima
        ]
        
        # Tenta mover na direção atual da espiral
        dx, dy = spiral_directions[self.spiral_direction]
        new_x = self.x + dx
        new_y = self.y + dy
        new_pos = (new_x, new_y)
        
        # Verifica se a nova posição é válida
        if (x_min <= new_x < x_max and y_min <= new_y < y_max and 
            obstacles[self.spiral_direction] == VS.CLEAR and 
            new_pos not in self.visited):
            
            self.spiral_count += 1
            
            # Se completou o passo atual, muda de direção
            if self.spiral_count >= self.spiral_step:
                self.spiral_count = 0
                # Usa a sequência de direções específica deste explorador
                current_index = self.direction_sequence.index(self.spiral_direction)
                next_index = (current_index + 1) % 4
                self.spiral_direction = self.direction_sequence[next_index]
                
                # Se completou uma volta completa, aumenta o tamanho do passo
                if self.spiral_direction == self.direction_sequence[0]:
                    self.spiral_step += 1
                    self.spiral_turns += 1
                    
                    # Se já deu muitas voltas, começa a voltar
                    if self.spiral_turns > 3:
                        return None
            
            return (dx, dy)
        
        # Se não pode mover na direção atual, tenta outras direções na sequência
        for direction in self.direction_sequence:
            if obstacles[direction] == VS.CLEAR:
                dx, dy = spiral_directions[direction]
                new_x = self.x + dx
                new_y = self.y + dy
                new_pos = (new_x, new_y)
                
                if (x_min <= new_x < x_max and y_min <= new_y < y_max and 
                    new_pos not in self.visited):
                    return (dx, dy)
        
        # Se não encontrou nenhuma direção válida, tenta qualquer direção possível
        for direction in range(8):
            if obstacles[direction] == VS.CLEAR:
                dx, dy = Explorer.AC_INCR[direction]
                new_x = self.x + dx
                new_y = self.y + dy
                new_pos = (new_x, new_y)
                
                if new_pos not in self.visited:
                    return (dx, dy)
        
        # Se não encontrou nenhuma direção válida, começa a voltar
        return None

    def explore(self):
        # get an random increment for x and y       
        next_pos = self.get_next_position()
        
        # Se não houver mais posições novas para explorar, começa a voltar
        if next_pos is None:
            return False
            
        dx, dy = next_pos

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Adiciona delay para visualização
        time.sleep(Explorer.DELAY)

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            
            # Adiciona a nova posição ao conjunto de visitados
            self.visited.add((self.x, self.y))

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

            # Se encontrou uma vítima, marca no mapa
            if seq != VS.NO_VICTIM:
                self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
                #print(f"{self.NAME} Victim {seq} marked at ({self.x}, {self.y})")

        return True

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        
        # Adiciona delay para visualização
        time.sleep(Explorer.DELAY)
        
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position
        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if self.walk_time < (self.get_rtime() - time_tolerance):
            # Se explore retornar False, significa que não há mais posições novas para explorar
            if not self.explore():
                # Se já estiver na base, termina
                if self.x == self.base_x and self.y == self.base_y:
                    # Garante que todas as vítimas encontradas estejam no mapa
                    for victim_id, (pos, vs) in self.victims.items():
                        self.map.add(pos, 1, victim_id, self.check_walls_and_lim())
                    self.resc.sync_explorers(self.map, self.victims)
                    return False
                # Caso contrário, volta para a base
                self.come_back()
            return True

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == self.base_x and self.y == self.base_y):
            # Garante que todas as vítimas encontradas estejam no mapa
            for victim_id, (pos, vs) in self.victims.items():
                self.map.add(pos, 1, victim_id, self.check_walls_and_lim())
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

