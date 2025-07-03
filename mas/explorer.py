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
    
    def __init__(self, env, config_file, resc, initial_direction=None, region_min_y=None, region_max_y=None):
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
        print(f"DEBUG: {self.NAME} inicializado")

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        
        # Inicializa a fronteira com as posições adjacentes à base
        obstacles = self.check_walls_and_lim()
        
        # Se tiver uma direção inicial, prioriza ela
        if self.initial_direction is not None and obstacles[self.initial_direction] == VS.CLEAR:
            dx, dy = self.AC_INCR[self.initial_direction]
            nx, ny = self.x + dx, self.y + dy
            self.frontier.append((nx, ny, [(dx, dy)]))
            print(f"DEBUG: {self.NAME} direção inicial {self.initial_direction} adicionada à fronteira")
        
        # Adiciona as outras direções possíveis
        for i, (dx, dy) in self.AC_INCR.items():
            if i != self.initial_direction and obstacles[i] == VS.CLEAR:
                nx, ny = self.x + dx, self.y + dy
                self.frontier.append((nx, ny, [(dx, dy)]))
        print(f"DEBUG: {self.NAME} inicialização - fronteira: {len(self.frontier)} posições")

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

            # Limpa a fronteira de posições já visitadas
            cleaned_frontier = deque()
            for item in self.frontier:
                next_x, next_y, path = item
                if (next_x, next_y) not in self.visited:
                    cleaned_frontier.append(item)
            self.frontier = cleaned_frontier
            
            if not self.frontier:
                # Se não há mais posições válidas na fronteira
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
        print(f"DEBUG: {self.NAME} explore chamado - posição atual: ({self.x},{self.y}) - fronteira: {list(self.frontier)}")
        next_move = self.get_next_position()
        if next_move is None:
            print(f"DEBUG: {self.NAME} explore - next_move é None")
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
                    # Verifica se a posição já está na fronteira para evitar duplicatas
                    already_in_frontier = False
                    for item in self.frontier:
                        if item[0] == new_x and item[1] == new_y:
                            already_in_frontier = True
                            break
                    
                    if not already_in_frontier:
                        new_path = [(dx, dy)]
                        self.frontier.append((new_x, new_y, new_path))
                        self.unexplored_cells.add((new_x, new_y))

    def come_back(self):
        print(f"DEBUG: {self.NAME} tentando voltar para base de ({self.x},{self.y})")
        if not self.base_path:
            print(f"DEBUG: {self.NAME} calculando novo caminho para base")
            # Se não tiver caminho para a base, calcula um novo
            self.calculate_path_to_base()
        if self.base_path:
            print(f"DEBUG: {self.NAME} base_path: {self.base_path}")
            dx, dy = self.base_path.pop(0)
            print(f"DEBUG: {self.NAME} vai andar dx={dx}, dy={dy}")
            result = self.walk(dx, dy)
            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy
                print(f"DEBUG: {self.NAME} moveu para ({self.x},{self.y})")
            else:
                print(f"DEBUG: {self.NAME} não conseguiu andar, recalculando caminho")
                self.base_path = []
        else:
            print(f"DEBUG: {self.NAME} não conseguiu calcular caminho para base")
            
        # Se chegou na base, chama sync_explorers e finaliza
        if self.x == 0 and self.y == 0:
            print(f"DEBUG: {self.NAME} chegou na base e vai chamar sync_explorers")
            self.resc.sync_explorers(self.map, self.victims)
            print(f"DEBUG: {self.NAME} chamou sync_explorers")
            self.set_state(VS.ENDED)
            return False

    def estimate_time_to_base(self):
        """ Estima o tempo necessário para voltar para a base """
        manhattan_dist = abs(self.x) + abs(self.y)
        estimated_time = manhattan_dist * self.COST_LINE
        return estimated_time

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
        print(f"DEBUG: {self.NAME} deliberate início - posição ({self.x},{self.y}) - walk_time={self.walk_time} - rtime={self.get_rtime()}")
        remaining_time = self.get_rtime()
        time_to_return = self.estimate_time_to_base()
        print(f"DEBUG: {self.NAME} remaining_time={remaining_time}, time_to_return={time_to_return}")
        
        # Se a fronteira estiver vazia ou walk_time > 4000, volta para base
        if not self.frontier or self.walk_time > 4000:
            print(f"DEBUG: {self.NAME} fronteira vazia ou walk_time alto, vai voltar para base")
            self.come_back()
            return True
        
        # Contador de segurança: se explorou muito tempo sem encontrar vítimas, volta
        if self.walk_time > 3000 and len(self.victims) == 0:
            print(f"DEBUG: {self.NAME} explorou muito tempo sem encontrar vítimas, vai voltar para base")
            self.come_back()
            return True
        
        # Ser mais agressivo na exploração - só volta se realmente não tiver tempo
        if remaining_time > time_to_return + 100:
            print(f"DEBUG: {self.NAME} explorando (vai explorar)")
            self.explore()
            return True
        # Se não pode mais explorar, tenta voltar para base
        print(f"DEBUG: {self.NAME} voltando para base (vai voltar)")
        self.come_back()
        return True

    def walk(self, dx, dy):
        print(f"DEBUG: {self.NAME} walk chamado - posição ({self.x},{self.y}), dx={dx}, dy={dy}, walk_time antes={self.walk_time}")
        result = super().walk(dx, dy)
        # Incrementa o tempo de caminhada
        if result == VS.EXECUTED:
            if abs(dx) == 1 and abs(dy) == 1:
                self.walk_time += self.COST_DIAG
            else:
                self.walk_time += self.COST_LINE
            print(f"DEBUG: {self.NAME} walk_time depois={self.walk_time}")
        return result

