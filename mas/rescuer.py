##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
import pickle
import numpy as np
import heapq as hp
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    # Variáveis de classe para o classificador e scaler
    classifier = None
    scaler = None
    model_loaded = False

    @classmethod
    def load_model(cls):
        if not cls.model_loaded:
            model_path = os.path.join(os.path.dirname(__file__), "models", "victim_classifier.pkl")
            scaler_path = os.path.join(os.path.dirname(__file__), "models", "victim_scaler.pkl")
            try:
                with open(model_path, 'rb') as f:
                    cls.classifier = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    cls.scaler = pickle.load(f)
                cls.model_loaded = True
                print("[Rescuer] Classificador e scaler carregados com sucesso!")
            except Exception as e:
                print(f"[Rescuer] ERRO ao carregar o classificador/scaler: {e}")
                cls.classifier = None
                cls.scaler = None

    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited 
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster 
        
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        filename = os.path.join(os.path.dirname(__file__), "clusters", f"cluster{cluster_id}.txt")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = os.path.join(os.path.dirname(__file__), "clusters", f"seq{sequence_id}.txt")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """ this method clusters the victims using the K-Means algorithm
        
        @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
                  such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
                  including the severity value and the corresponding label"""

        vic = self.victims
        
        # Se não há vítimas, retorna lista vazia
        if not vic:
            return []
            
        # Determina o número de clusters baseado no número de vítimas
        num_victims = len(vic)
        num_clusters = min(4, num_victims)  # Máximo 4 clusters, mas não mais que o número de vítimas
        
        if num_clusters == 0:
            return []

        #randomize centroids to start up K-Means algorithm
        # Converte dict_items para lista antes de usar random.sample
        vic_items = list(vic.items())
        centroids = dict(random.sample(vic_items, num_clusters))

        #print("dictionary: ", vic, "\n") #Debugging
        #print("chosen centroids: ", centroids, "\n")

        cluster_changed = True
        number_of_iterations = 4
        i = 0
        
    
        # Divide dictionary into clusters - cria clusters dinamicamente
        clusters = [{} for _ in range(num_clusters)]
        
        while (i < number_of_iterations and cluster_changed == True): #Outer loop
            cluster_changed = False
            
            # Limpa os clusters para a nova iteração
            for cluster in clusters:
                cluster.clear()

            for key, values in vic.items():  # values are pairs: ((x,y), [<vital signals list>]), this loop assigns a cluster to each victim
                
                x, y = values[0]
                min_distance = 1000000000000
                min_cluster_idx = 0

                for c_key, c_values in centroids.items(): #determines the centroid closest to the current victim
                    
                    c_x, c_y = c_values[0]

                    delta_x, delta_y = c_x - x, c_y - y
                    distance = math.sqrt(delta_x**2 + delta_y**2)

                    if(distance < min_distance):
                        min_distance = distance
                        min_key = c_key
                    
                centroid_keys = list(centroids.keys())

                #Assigns each victim to a cluster based on which centroid they're closest by
                for idx, centroid_key in enumerate(centroid_keys):
                    if min_key == centroid_key:
                        clusters[idx][key] = values
                        break

            #print(f"Clusters from iteraction:{i}\n") #Debugging
            #print(f"\n{clusters}")

            j = 0 #tracks in which cluster/centroid we are

            for clusterX in clusters: # Updates each centroid's centers for next iteration, sees if at least one centroid was changed
                if len(clusterX) == 0:  # Skip empty clusters
                    j += 1
                    continue
                    
                sum_x = 0
                sum_y = 0
                current_key = centroid_keys[j]

                for values in clusterX.values():
                    x, y = values[0]
                    sum_x += x
                    sum_y += y

                c_mean = (sum_x/len(clusterX), sum_y/len(clusterX))
                #print(f"New c_mean for cluster{j}: {c_mean}") #Debugging

                
                if(c_mean != centroids[current_key][0]):
                    centroids.update({current_key: (c_mean, centroids[current_key][1])})
                    cluster_changed = True

                j += 1
            
            #print(f"New centroids: \n{centroids}\n") #Debugging
            i += 1

        return clusters

    
    def predict_severity_and_class(self):
        """ Usa o classificador treinado para prever a classe de severidade das vítimas. """
        # Carrega o modelo se ainda não foi carregado
        self.__class__.load_model()
        if self.__class__.classifier is None or self.__class__.scaler is None:
            print("[Rescuer] Classificador ou scaler não carregado. Usando valores aleatórios como fallback.")
            for vic_id, values in self.victims.items():
                severity_value = random.uniform(0.1, 99.9)
                severity_class = random.randint(1, 4)
                values[1].extend([severity_value, severity_class])
            return

        for vic_id, values in self.victims.items():
            # Sinais vitais: pSist, pDiast, qPA, pulso, resp
            features = np.array(values[1][:5]).reshape(1, -1)
            features_scaled = self.__class__.scaler.transform(features)
            predicted_class = int(self.__class__.classifier.predict(features_scaled)[0])
            # Valor de gravidade: manter aleatório (ou pode-se usar um regressor no futuro)
            severity_value = random.uniform(0.1, 99.9)
            values[1].extend([severity_value, predicted_class])


    @staticmethod
    def heuristic(coord1, coord2, min_difficulty):
        """ A heuristic function to estimate the cost between two coordinates.
            It uses the Diagonal Distance, adjusted by the minimum difficulty of the map.
            @param coord1: first coordinate (x1, y1)
            @param coord2: second coordinate (x2, y2)
            @return: estimated cost between coord1 and coord2 """
        x1, y1 = coord1
        x2, y2 = coord2
        deltaX = abs(x2 - x1)
        deltaY = abs(y2 - y1)
        distance = deltaX + deltaY - min(deltaX, deltaY)* 0.5 # You save 0.5 for all diagonal moves
        return distance*min_difficulty

    
    @staticmethod
    def a_star(map, coord_start, coord_goal, min_difficulty):
        """ A* search function, finds the shortest distance between two coordinates on the map.
            @param coord_start: start coordinate (x1, y1)
            @param coord_goal: goal coordinate (x2, y2)
            @param min_difficulty: The lowest value of acess difficulty
            @return: estimated cost between coord1 and coord2 """
        
        #Set starting values for algorithm
        coord_difficulty = map.get_difficulty(coord_start)
        

        priority_queue = []
        done_queue = []
        dist_traveled = 0
        
        #this is the initial node, each node of the heap is a tuple in the format:(estimated total distance to goal, coordinate, difficulty, minimum distance traveled)
        hp.heappush(priority_queue, (Rescuer.heuristic(coord_start, coord_goal, min_difficulty),coord_start, coord_difficulty, dist_traveled))

        while(priority_queue):

            current = hp.heappop(priority_queue)
            done_queue.append(current[1])

            actions = map.get_actions_results(current[1])

            neighbors = []
            

            for i in range(len(actions)):
                if(actions[i] == VS.CLEAR): #if the action in direction can be taken

                    x, y = current[1]
                    movement_cost = 1

                    match i: #adjusts coordinate and movement cost, for target of action, based on its position in the actions list
                        case 0:
                            y += 1
                        case 1:
                            x += 1
                            y += 1
                            movement_cost = 1.5
                        case 2:
                            x += 1
                        case 3:
                            x += 1
                            y -= 1
                            movement_cost = 1.5
                        case 4:
                            y -= 1
                        case 5:
                            y -= 1
                            x -= 1
                            movement_cost = 1.5
                        case 6:
                            x -= 1
                        case 7:
                            x -=1
                            y += 1
                            movement_cost = 1.5

                    neighbor_coord = x, y
                    if(neighbor_coord in done_queue): # skips node if it has already been reached by the best route
                        continue
                    
                    neighbor_difficulty = map.get_difficulty(neighbor_coord)
                    dist_to_neighbor = current[3] + movement_cost * neighbor_difficulty
                    estimated_distance = Rescuer.heuristic(neighbor_coord, coord_goal, min_difficulty) + dist_to_neighbor


                    if(neighbor_coord not in [pq_list[1] for pq_list in priority_queue]):
                        if(neighbor_coord == coord_goal): 
                            return dist_to_neighbor # Return minimum distance traveled from start to goal
                            
                        hp.heappush(priority_queue, (estimated_distance, neighbor_coord, neighbor_difficulty, dist_to_neighbor))
                    else: #neighbor already discovered
                        j = 0
                        while(j < len(priority_queue)):
                            #Searches for the node in the priority queue e se a distância encontrada for menor, substitui a tupla inteira
                            if(priority_queue[j][1] == neighbor_coord and priority_queue[j][0] > estimated_distance):
                                priority_queue[j] = (estimated_distance, neighbor_coord, neighbor_difficulty, dist_to_neighbor)
                                break #found the node, stop the search
                            j += 1
                            
    @staticmethod
    def fitness(sequence, distance_matrix):
        """ fitness function, calculates total lenght of a sequence
            @param sequence: a sequence of victims to visit, 
            @param distance_matrix: a matrix storing the distance between victim x and y in m[x][y], victim 0 represents the starting point.
            @return: the total lenght of given sequence"""
        dist_sum = 0
        
        for i in range(len(sequence) - 1):
            dist_sum += distance_matrix[sequence[i]][sequence[i + 1]]
        
        return dist_sum
        
            
            
                            
    @staticmethod
    def genetic_algorithm(indexed_victims, distance_matrix, pop_size, generations, cx_prob, mut_prob):
        """ genetic algorithm, finds the best combination of victims to rescue, based on maximizing victim visits in the shortest time.
            utilizing tourney for selection
            @param indexed_victims: a dictionary of victims in the format: {index : vic_id, (x, y), ["vs"]}
            @param distance_matrix: a matrix storing the distance between victim x and y in m[x][y], victim 0 represents the starting point.
            @param pop_size: the number of individuals per population
            @param generations: the number of generations for simulation
            @param cx_prob: probability of crossover, in interval ]0, 1]
            @param mut_prob: probability of mutation, in interval ]0, 1]
            @return: best sequence of victims (whithout starting point) in the format:{vic_id : (x, y), ["vs"]}"""
            
        starting_population = []
        vic_number = len(indexed_victims)
        
        # Se há apenas uma vítima (além do ponto inicial), retorna a sequência original
        if vic_number <= 2:
            best_sequence = {}
            for index in range(1, vic_number):
                values = indexed_victims.get(index)
                if values:
                    id, pos, condition = values
                    best_sequence[id] = (pos, condition)
            return best_sequence
            
        #populates generation 0 randomly
        for _ in range(pop_size):
            random_vic_order = list(indexed_victims.keys())
            random.shuffle(random_vic_order)
              
            #victim 0 must be in first position, since it is the starting position
            for i in range(vic_number):
                if(random_vic_order[i] == 0):
                    aux = random_vic_order[0]
                    random_vic_order[0] = random_vic_order[i]
                    random_vic_order[i] = aux
                    break
                    
            starting_population.append(random_vic_order)
            
        for gen in range(generations):
            parent_population = []
            new_population = []
                
            #Determines the parent generation via tournament, each parent is the best of 3 random individuals from previous generation
            for i in range(pop_size):
                if len(starting_population) >= 3:
                    tourney_candidates = random.sample(starting_population, 3)
                else:
                    tourney_candidates = starting_population
                min_fitness = 10000
                #finds best candidate
                for candidate in tourney_candidates:
                    fit_test = Rescuer.fitness(candidate, distance_matrix)
                        
                    if(fit_test < min_fitness):
                        winner = candidate
                        min_fitness = fit_test

                            
                parent_population.append(winner)
                    
                #every other time a parent gets added
                if(i%2 == 1 and i < len(parent_population)):
                       
                    if i - 1 < len(parent_population) and i < len(parent_population):
                        parent_1 = parent_population[i - 1]
                        parent_2 = parent_population[i]
                    else:
                        continue

                    #parental crossover over gene strip
                    if(random.random() <= cx_prob):
                    
                        child_1 = [-1]*vic_number
                        child_2 = [-1]*vic_number
                        
                        child_1[0] = parent_1[0]
                        child_2[0] = parent_2[0]
                        
                        #gene strip from start to end
                        if vic_number > 2:
                            start, end = sorted(random.sample(range(1, vic_number), 2))
                        else:
                            start, end = 1, 1
                        child_1[start:end] = parent_1[start:end]
                        child_2[start:end] = parent_2[start:end]
                            
                        if vic_number > 2:
                            iterator = [j for j in range(1, start - 1)] + [j for j in range(end + 1, vic_number)]
                            
                            #Crossover genes
                            fill_genes_1 = [gene for gene in parent_1 if gene not in child_2]
                            fill_genes_2 = [gene for gene in parent_2 if gene not in child_1]
                            k = 0
                                
                            for j in iterator:
                                if k < len(fill_genes_2) and k < len(fill_genes_1):
                                    child_1[j] = fill_genes_2[k]
                                    child_2[j] = fill_genes_1[k]
                                    k += 1
                    else: #No crossover
                        child_1 = parent_1
                        child_2 = parent_2
                        
                    #Possible mutation of first child
                    if(random.random() < mut_prob and vic_number > 2):
                        gene1, gene2 = random.sample(range(1, vic_number), 2)
                        child_1[gene1], child_1[gene2] = child_1[gene2], child_2[gene1]
                    #Possible mutation of second child
                    if(random.random() < mut_prob and vic_number > 2):
                        gene1, gene2 = random.sample(range(1, vic_number), 2)
                        child_2[gene1], child_2[gene2] = child_2[gene2], child_2[gene1]
                    
                    new_population.append(child_1)
                    new_population.append(child_2)
            
            #Setup for next generation on next iteration
            starting_population = new_population 
        

        #Formatting return value, removing indexes from dictionary
        best_sequence = {}
        if new_population:
            best_individual = new_population[0]  # Take the first individual as the best
            for index in best_individual[1:vic_number]:
                values = indexed_victims.get(index)
                if values:
                    id, pos, condition = values
                    best_sequence[id] = (pos, condition)
        else:
            # Se não há população, retorna a sequência original
            for index in range(1, vic_number):
                values = indexed_victims.get(index)
                if values:
                    id, pos, condition = values
                    best_sequence[id] = (pos, condition)
        
        return best_sequence
        
    def sequencing(self):
        """ This method relies on a Genetic Algorithm to find the possibly best visiting order"""

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""

        #original placeholder definition:

        #new_sequences = []

        #for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            #seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            #new_sequences.append(seq)       
            #print(f"{self.NAME} sequence of visit:\n{seq}\n")

        #self.sequences = new_sequences


        new_sequences = []
        map = self.map

        # get minimum difficulty for each coordinate in map
        # the minimum difficulty is used in the heuristic of the A* algorithm
        # to find the shortest path between victims

        min_difficulty = 100
        x, y = 0, 0

        while map.in_map((x, y)): #goes through x values

            while map.in_map((x, y)): #goes through y values

                difficulty = map.get_difficulty((x, y))
                if difficulty < min_difficulty:
                    min_difficulty = difficulty
                y += 1

            y = 0
            x += 1


        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary

            # Corrigir a montagem do indexed_victims para garantir que (x, y) seja tupla
            indexed_victims = {}
            idx = 1
            for vic_id, values in seq.items():
                # Garantir que values[0] seja tupla
                coord = tuple(values[0])
                indexed_victims[idx] = (vic_id, coord, values[1])
                idx += 1
            indexed_victims[0] = ("", (self.plan_x, self.plan_y), "")
            num_victims = len(indexed_victims)

            distance_matrix = [[-1 for _ in range(num_victims)] for _ in range(num_victims)]
            
            #fills distance_matrix, each position (x, y) in the matrix is the minimum distance between victims x and y in the map.
            for i in range(num_victims):
                for j in range(i):
                    distance_matrix[i][j] = Rescuer.a_star(map, indexed_victims.get(i)[1], indexed_victims.get(j)[1], min_difficulty)
                    distance_matrix[j][i] = distance_matrix[i][j]
            
            
            new_sequences.append(self.genetic_algorithm(indexed_victims, distance_matrix, 100, 50, 0.8, 0.2))
        
        self.sequences = new_sequences

    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""


        # let's instantiate the breadth-first search
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        start = (0,0) # always from starting at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal

        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        self.plan = self.plan + plan
        self.plan_rtime = self.plan_rtime - time
           

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #self.map.draw()
            #print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()
            
            # Se não há clusters, não há nada para fazer
            if not clusters_of_vic:
                print(f"{self.NAME} No victims found to cluster")
                return

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1
  
            # Determina quantos rescuers precisamos baseado no número de clusters
            num_clusters = len(clusters_of_vic)
            num_rescuers = min(4, num_clusters)  # Máximo 4 rescuers
            
            # Instantiate the other rescuers
            rescuers = [None] * num_rescuers
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of 
            self.clusters = [clusters_of_vic[0]]  # the first one

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, num_rescuers):    
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map     # each rescuer have the map

            
            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters         

            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()         # the sequencing will reorder the cluster
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1 
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

            
                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True

