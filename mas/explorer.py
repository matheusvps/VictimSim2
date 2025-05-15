# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from collections import deque

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
    
    def __init__(self, env, config_file, resc, initial_direction=None):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        @param initial_direction: direção inicial do explorador (0-7)
        """

        super().__init__(env, config_file)
        self.walk_time = 0         # time consumed to walk when exploring
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
        self.frontier = deque()    # fila para busca em largura
        self.visited = set()       # conjunto de posições visitadas
        self.current_path = []     # caminho atual sendo seguido
        self.base_path = []        # caminho de volta para a base
        self.initial_direction = initial_direction  # direção inicial do explorador
        self.unexplored_cells = set()  # células que ainda não foram exploradas
        self.last_position = (0, 0)    # última posição visitada
        self.heat_map = {}        # mapa de calor para rastrear áreas promissoras
        self.victim_clusters = []  # clusters de vítimas encontradas
        self.exploration_radius = 5  # raio inicial de exploração
        self.max_radius = 20      # raio máximo de exploração

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        
        # Inicializa a fronteira com as posições adjacentes à base
        obstacles = self.check_walls_and_lim()
        
        # Se tiver uma direção inicial, prioriza ela
        if self.initial_direction is not None and obstacles[self.initial_direction] == VS.CLEAR:
            dx, dy = self.AC_INCR[self.initial_direction]
            self.frontier.append((self.x + dx, self.y + dy, [(dx, dy)]))
        
        # Adiciona as outras direções possíveis
        for i, (dx, dy) in self.AC_INCR.items():
            if i != self.initial_direction and obstacles[i] == VS.CLEAR:
                self.frontier.append((self.x + dx, self.y + dy, [(dx, dy)]))

    def update_heat_map(self, x, y, has_victim):
        """ Atualiza o mapa de calor baseado em vítimas encontradas """
        # Atualiza a célula atual
        if has_victim:
            self.heat_map[(x, y)] = self.heat_map.get((x, y), 0) + 2.0
        else:
            self.heat_map[(x, y)] = self.heat_map.get((x, y), 0) - 0.1

        # Atualiza células vizinhas
        for dx, dy in self.AC_INCR.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.visited:
                if has_victim:
                    self.heat_map[(nx, ny)] = self.heat_map.get((nx, ny), 0) + 0.5
                else:
                    self.heat_map[(nx, ny)] = self.heat_map.get((nx, ny), 0) - 0.05

    def find_victim_clusters(self):
        """ Identifica clusters de vítimas para focar a exploração """
        clusters = []
        visited = set()

        for pos, victim_data in self.victims.items():
            if pos in visited:
                continue

            cluster = set()
            queue = deque([pos])
            visited.add(pos)

            while queue:
                x, y = queue.popleft()
                cluster.add((x, y))

                for dx, dy in self.AC_INCR.values():
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in self.victims and (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))

            if cluster:
                clusters.append(cluster)

        return clusters

    def get_next_position(self):
        """ Obtém a próxima posição usando busca em largura com priorização por heat map """
        if not self.current_path:
            if not self.frontier:
                # Procura células não exploradas com maior probabilidade
                next_pos = self.find_best_unexplored_cell()
                if next_pos:
                    self.current_path = self.calculate_path_to_position(next_pos)
                    return self.current_path[0] if self.current_path else None
                return None

            # Ordena a fronteira pelo heat map
            frontier_list = list(self.frontier)
            frontier_list.sort(key=lambda x: self.heat_map.get((x[0], x[1]), 0), reverse=True)
            self.frontier = deque(frontier_list)

            next_x, next_y, path = self.frontier.popleft()

            if (next_x, next_y) in self.visited:
                return self.get_next_position()

            self.visited.add((next_x, next_y))
            self.current_path = path
            return self.current_path[0]

        return self.current_path.pop(0)

    def find_best_unexplored_cell(self):
        """ Encontra a melhor célula não explorada baseada no heat map e distância """
        if not self.unexplored_cells:
            return None

        best_score = float('-inf')
        best_cell = None

        for cell in self.unexplored_cells:
            heat = self.heat_map.get(cell, 0)
            dist = math.sqrt((cell[0] - self.x)**2 + (cell[1] - self.y)**2)
            score = heat - (dist * 0.1)  # penaliza distância

            if score > best_score:
                best_score = score
                best_cell = cell

        return best_cell

    def calculate_path_to_position(self, target_pos):
        """ Calcula o caminho mais curto para uma posição específica usando BFS """
        queue = deque([(self.x, self.y, [])])
        visited = {(self.x, self.y)}
        
        while queue:
            x, y, path = queue.popleft()
            
            if (x, y) == target_pos:  # Chegou na posição alvo
                return path
                
            # Ordena as direções pela probabilidade de vítimas
            directions = []
            for i, (dx, dy) in self.AC_INCR.items():
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) not in visited:
                    cell = self.map.get((new_x, new_y))
                    if cell and cell[0] != VS.OBST_WALL:
                        prob = self.heat_map.get((new_x, new_y), 0.1)
                        directions.append((i, prob))
            
            # Ordena por probabilidade decrescente
            directions.sort(key=lambda x: x[1], reverse=True)
            
            for i, _ in directions:
                dx, dy = self.AC_INCR[i]
                new_x, new_y = x + dx, y + dy
                visited.add((new_x, new_y))
                queue.append((new_x, new_y, path + [(dx, dy)]))
        
        return None

    def explore(self):
        next_move = self.get_next_position()
        
        if next_move is None:
            return
            
        dx, dy = next_move

        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.BUMPED:
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

        if result == VS.EXECUTED:
            self.x += dx
            self.y += dy
            self.last_position = (self.x, self.y)
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)

            seq = self.check_for_victim()
            has_victim = seq != VS.NO_VICTIM
            if has_victim:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                self.update_heat_map(self.x, self.y, True)
            else:
                self.update_heat_map(self.x, self.y, False)

            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

            obstacles = self.check_walls_and_lim()
            for i, (dx, dy) in self.AC_INCR.items():
                new_x = self.x + dx
                new_y = self.y + dy
                if obstacles[i] == VS.CLEAR and (new_x, new_y) not in self.visited:
                    new_path = [(dx, dy)]
                    self.frontier.append((new_x, new_y, new_path))
                    self.unexplored_cells.add((new_x, new_y))

    def come_back(self):
        """ Volta para a base usando o caminho mais curto """
        if not self.base_path:
            # Se não tiver caminho para a base, calcula um novo
            self.calculate_path_to_base()
            
        if self.base_path:
            dx, dy = self.base_path.pop(0)
            result = self.walk(dx, dy)
            
            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy

    def calculate_path_to_base(self):
        """ Calcula o caminho mais curto para a base usando BFS """
        queue = deque([(self.x, self.y, [])])
        visited = {(self.x, self.y)}
        
        while queue:
            x, y, path = queue.popleft()
            
            if x == 0 and y == 0:  # Chegou na base
                self.base_path = path
                return
                
            for i, (dx, dy) in self.AC_INCR.items():
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) not in visited:
                    cell = self.map.get((new_x, new_y))
                    if cell and cell[0] != VS.OBST_WALL:  # Se não for parede
                        visited.add((new_x, new_y))
                        queue.append((new_x, new_y, path + [(dx, dy)]))
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # Se ainda tiver tempo para explorar
        if self.walk_time < (self.get_rtime() - 2*self.COST_DIAG):
            self.explore()
            return True

        # Se já estiver na base
        if self.x == 0 and self.y == 0:
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

