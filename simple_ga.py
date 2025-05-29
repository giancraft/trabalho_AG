import numpy as np
import random
from copy import deepcopy
from representation import generate_tree, simplify_tree, count_nodes
from fitness import fitness, optimize_constants
from operators import select_parents, crossover, mutate, mutate_constants

def generate_individual(max_depth):
    """Gera um indivíduo (árvore + constantes)"""
    tree = generate_tree(max_depth)
    constants = [random.uniform(-10, 10) for _ in range(10)]
    return (tree, constants)

def genetic_algorithm(X, y, pop_size=100, max_depth=4, generations=50, 
                      cx_prob=0.7, mut_prob=0.2, num_elites=1, 
                      optimize_constants_every=5):
    # Inicializar população com indivíduos (árvore + constantes)
    population = [generate_individual(max_depth) for _ in range(pop_size)]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(generations):
        # Otimização de constantes (em intervalos especificados)
        if gen % optimize_constants_every == 0:
            optimized_population = []
            for tree, constants in population:
                optimized_constants = optimize_constants(tree, X, y, constants)
                optimized_population.append((tree, optimized_constants))
            population = optimized_population
        
        # Calcular fitness
        fitnesses = [fitness(ind, X, y) for ind in population]
        
        if not fitnesses: 
            break 

        best_fitness_gen = min(fitnesses) if fitnesses else float('inf')
        avg_fitness_gen = sum(fitnesses) / len(fitnesses) if fitnesses else float('inf')
        best_fitness_history.append(best_fitness_gen)
        avg_fitness_history.append(avg_fitness_gen)
        
        # print(f"Geração {gen+1}/{generations} - Melhor Fitness: {best_fitness_gen:.4f}, Fitness Médio: {avg_fitness_gen:.4f}")

        new_population = []

        # Elitismo: selecionar os melhores indivíduos
        if num_elites > 0 and population:
            elite_indices = np.argsort(fitnesses)[:num_elites]
            elites = [deepcopy(population[i]) for i in elite_indices]
            new_population.extend(elites)

        num_offspring_needed = pop_size - len(new_population)
        current_offspring_count = 0
        
        while current_offspring_count < num_offspring_needed:
            if len(population) < 2: 
                new_population.append(generate_individual(max_depth))
                current_offspring_count += 1
                continue

            parents = select_parents(population, fitnesses)
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
        
        population = new_population[:pop_size] 
    
    # Última otimização de constantes
    optimized_population = []
    for tree, constants in population:
        optimized_constants = optimize_constants(tree, X, y, constants)
        optimized_population.append((tree, optimized_constants))
    population = optimized_population
    
    # Avaliação final
    final_fitnesses = [fitness(ind, X, y) for ind in population]
    if not final_fitnesses: 
        return None, float('inf'), best_fitness_history, avg_fitness_history

    best_idx = np.argmin(final_fitnesses)
    best_individual = population[best_idx]
    best_fitness_val = final_fitnesses[best_idx]
    
    return best_individual, best_fitness_val, best_fitness_history, avg_fitness_history