import random
import numpy as np
from copy import deepcopy
from representation import generate_tree
from fitness import fitness
from operators import select_parents, crossover, mutate

def island_genetic_algorithm(X, y, num_islands=4, pop_per_island=50, 
                             max_depth=4, generations=100, cx_prob=0.7, 
                             mut_prob=0.2, migration_interval=10, 
                             migration_rate=0.1):
    islands = []
    constants = [[random.uniform(-10, 10) for _ in range(10)] for _ in range(num_islands)]
    
    for _ in range(num_islands):
        island_pop = [generate_tree(max_depth) for _ in range(pop_per_island)]
        islands.append(island_pop)
    
    best_fitness_history = [[] for _ in range(num_islands)]
    avg_fitness_history = [[] for _ in range(num_islands)]
    
    for gen in range(generations):
        for i in range(num_islands):
            fitnesses = [fitness(ind, X, y, constants[i]) for ind in islands[i]]
            
            best_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness_history[i].append(best_fitness)
            avg_fitness_history[i].append(avg_fitness)
            
            new_population = []
            while len(new_population) < pop_per_island:
                parents = select_parents(islands[i], fitnesses)
                offspring1, offspring2 = deepcopy(parents[0]), deepcopy(parents[1])
                
                if random.random() < cx_prob:
                    try:
                        offspring1, offspring2 = crossover(offspring1, offspring2)
                    except:
                        pass
                
                if random.random() < mut_prob:
                    offspring1 = mutate(offspring1, max_depth)
                if random.random() < mut_prob:
                    offspring2 = mutate(offspring2, max_depth)
                
                new_population.extend([offspring1, offspring2])
            
            islands[i] = new_population[:pop_per_island]
        
        if gen % migration_interval == 0 and gen > 0:
            for i in range(num_islands):
                next_island = (i + 1) % num_islands
                
                num_emigrants = int(migration_rate * pop_per_island)
                emigrants = random.sample(islands[i], num_emigrants)
                
                for emigrant in emigrants:
                    idx = random.randint(0, pop_per_island - 1)
                    islands[next_island][idx] = deepcopy(emigrant)
    
    best_global = None
    best_global_fitness = float('inf')
    for i in range(num_islands):
        fitnesses = [fitness(ind, X, y, constants[i]) for ind in islands[i]]
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_global_fitness:
            best_global = islands[i][best_idx]
            best_global_fitness = fitnesses[best_idx]
    
    return best_global, best_global_fitness, best_fitness_history, avg_fitness_history