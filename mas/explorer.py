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

    def get_next_position(self):
        """ Obtém a próxima posição usando busca em largura
            Retorna None se não houver mais posições para explorar
        """
        # Se não houver caminho atual sendo seguido
        if not self.current_path:
            # Se a fronteira estiver vazia, não há mais posições para explorar
            if not self.frontier:
                return None
                
            # Pega a próxima posição da fronteira
            next_x, next_y, path = self.frontier.popleft()
            
            # Se a posição já foi visitada, pula
            if (next_x, next_y) in self.visited:
                return self.get_next_position()
            
            # Marca como visitada
            self.visited.add((next_x, next_y))
            
            # Define o caminho atual
            self.current_path = path
            
            # Retorna o primeiro movimento do caminho
            return self.current_path[0]
            
        # Se já estiver seguindo um caminho, retorna o próximo movimento
        return self.current_path.pop(0)

    def explore(self):
        # get next position using BFS
        next_move = self.get_next_position()
        
        if next_move is None:
            return
            
        dx, dy = next_move

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            
            # Adiciona as posições vizinhas à fronteira
            obstacles = self.check_walls_and_lim()
            for i, (dx, dy) in self.AC_INCR.items():
                if obstacles[i] == VS.CLEAR:
                    new_x = self.x + dx
                    new_y = self.y + dy
                    if (new_x, new_y) not in self.visited:
                        # Cria um novo caminho adicionando o movimento atual
                        new_path = [(dx, dy)]
                        self.frontier.append((new_x, new_y, new_path))

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

