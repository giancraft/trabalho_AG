import numpy as np
import random
from copy import deepcopy
from representation import generate_tree
from fitness import fitness
from operators import select_parents, crossover, mutate

def genetic_algorithm(X, y, pop_size=100, max_depth=4, generations=50, 
                      cx_prob=0.7, mut_prob=0.2):
    population = [generate_tree(max_depth) for _ in range(pop_size)]
    constants = [random.uniform(-10, 10) for _ in range(10)]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(generations):
        fitnesses = [fitness(ind, X, y, constants) for ind in population]
        
        best_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        new_population = []
        while len(new_population) < pop_size:
            parents = select_parents(population, fitnesses)
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
        
        population = new_population[:pop_size]
    
    fitnesses = [fitness(ind, X, y, constants) for ind in population]
    best_idx = np.argmin(fitnesses)
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    return best_individual, best_fitness, best_fitness_history, avg_fitness_history