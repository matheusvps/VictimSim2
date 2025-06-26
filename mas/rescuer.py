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


    def sequencing(self):
        """ Currently, this method sort the victims by the x coordinate followed by the y coordinate
            @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""

        new_sequences = []

        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            new_sequences.append(seq)       
            #print(f"{self.NAME} sequence of visit:\n{seq}\n")

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

