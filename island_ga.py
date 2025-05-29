import random
import numpy as np
from copy import deepcopy
from representation import generate_tree, simplify_tree, count_nodes
from fitness import fitness, optimize_constants
from operators import select_parents, crossover, mutate, mutate_constants

def generate_individual(max_depth):
    """Gera um indivíduo (árvore + constantes)"""
    tree = generate_tree(max_depth)
    constants = [random.uniform(-10, 10) for _ in range(10)]
    return (tree, constants)

def island_genetic_algorithm(X, y, num_islands=4, pop_per_island=50, 
                             max_depth=4, generations=100, cx_prob=0.7, 
                             mut_prob=0.2, migration_interval=10, 
                             migration_rate=0.1, num_elites=1,
                             optimize_constants_every=1):
    # Inicializar ilhas: cada ilha é uma lista de indivíduos (árvore, constantes)
    islands = []
    for _ in range(num_islands):
        island_pop = [generate_individual(max_depth) for _ in range(pop_per_island)]
        islands.append(island_pop)
    
    best_fitness_history = [[] for _ in range(num_islands)]
    avg_fitness_history = [[] for _ in range(num_islands)]
    
    for gen in range(generations):
        # Para cada ilha, processar
        for i in range(num_islands):
            # Otimização de constantes (em intervalos especificados)
            if gen % optimize_constants_every == 0:
                optimized_population = []
                for tree, constants in islands[i]:
                    optimized_constants = optimize_constants(tree, X, y, constants)
                    optimized_population.append((tree, optimized_constants))
                islands[i] = optimized_population
            
            # Calcular fitness
            fitnesses = [fitness(ind, X, y) for ind in islands[i]]
            
            if not fitnesses: 
                continue

            best_fitness_gen = min(fitnesses)
            avg_fitness_gen = sum(fitnesses) / len(fitnesses)
            best_fitness_history[i].append(best_fitness_gen)
            avg_fitness_history[i].append(avg_fitness_gen)
            
            # Elitismo: selecionar os melhores indivíduos
            new_population = []
            if num_elites > 0:
                elite_indices = np.argsort(fitnesses)[:num_elites]
                elites = [deepcopy(islands[i][idx]) for idx in elite_indices]
                new_population.extend(elites)
            
            num_offspring_needed = pop_per_island - len(new_population)
            current_offspring_count = 0
            
            while current_offspring_count < num_offspring_needed:
                if len(islands[i]) < 2: 
                    new_population.append(generate_individual(max_depth))
                    current_offspring_count += 1
                    continue

                parents = select_parents(islands[i], fitnesses)
                if parents[0] is None or parents[1] is None: 
                    new_population.append(generate_individual(max_depth))
                    current_offspring_count += 1
                    continue

                # Desempacotar pais
                parent1_tree, parent1_const = deepcopy(parents[0])
                parent2_tree, parent2_const = deepcopy(parents[1])
                
                # Gerar filhos
                offspring1_tree, offspring2_tree = deepcopy(parent1_tree), deepcopy(parent2_tree)
                offspring1_const, offspring2_const = deepcopy(parent1_const), deepcopy(parent2_const)
                
                # Crossover
                if random.random() < cx_prob:
                    try:
                        offspring1_tree, offspring2_tree = crossover(offspring1_tree, offspring2_tree)
                    except Exception:
                        pass 
                
                # Mutação de árvore
                if random.random() < mut_prob:
                    offspring1_tree = mutate(offspring1_tree, max_depth)
                if random.random() < mut_prob:
                    offspring2_tree = mutate(offspring2_tree, max_depth)
                    
                # Mutação de constantes
                offspring1 = mutate_constants((offspring1_tree, offspring1_const))
                offspring2 = mutate_constants((offspring2_tree, offspring2_const))
                
                # Simplificação das árvores
                offspring1_tree = simplify_tree(offspring1[0])
                offspring2_tree = simplify_tree(offspring2[0])
                offspring1 = (offspring1_tree, offspring1[1])
                offspring2 = (offspring2_tree, offspring2[1])
                
                # Adicionar filhos à nova população
                if current_offspring_count < num_offspring_needed:
                    new_population.append(offspring1)
                    current_offspring_count += 1
                if current_offspring_count < num_offspring_needed: 
                    new_population.append(offspring2)
                    current_offspring_count += 1
            
            islands[i] = new_population[:pop_per_island]
        
        # Migração entre ilhas
        if gen % migration_interval == 0 and gen > 0:
            for i in range(num_islands):
                next_island = (i + 1) % num_islands
                
                num_emigrants = int(migration_rate * pop_per_island)
                emigrants = random.sample(islands[i], num_emigrants)
                
                # Para cada emigrante, substituir um indivíduo aleatório na próxima ilha
                for emigrant in emigrants:
                    idx = random.randint(0, pop_per_island - 1)
                    islands[next_island][idx] = deepcopy(emigrant)
    
    # Última otimização de constantes
    for i in range(num_islands):
        optimized_population = []
        for tree, constants in islands[i]:
            optimized_constants = optimize_constants(tree, X, y, constants)
            optimized_population.append((tree, optimized_constants))
        islands[i] = optimized_population
    
    # Encontrar melhor indivíduo global
    best_global = None
    best_global_fitness = float('inf')
    for i in range(num_islands):
        fitnesses = [fitness(ind, X, y) for ind in islands[i]]
        if not fitnesses:
            continue
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_global_fitness:
            best_global = islands[i][best_idx]
            best_global_fitness = fitnesses[best_idx]
    
    return best_global, best_global_fitness, best_fitness_history, avg_fitness_history