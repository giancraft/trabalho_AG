import numpy as np
import random
import operator
import math
from copy import deepcopy
import time
import matplotlib.pyplot as plt

# ================================
# REPRESENTAÇÃO CROMOSSÔMICA
# ================================
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
    def __str__(self):
        """Representação em string para depuração"""
        if self.left is None and self.right is None:
            return self.value
        elif self.right is None:
            return f"{self.value}({self.left})"
        else:
            return f"({self.left} {self.value} {self.right})"

# Operadores e terminais
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda a, b: a / b if abs(b) > 1e-6 else 1.0,
    'sin': math.sin,
    'cos': math.cos
}

TERMINALS = ['x'] + [f'c{i}' for i in range(10)]

def safe_divide(a, b):
    """Divisão segura com proteção contra divisão por zero"""
    if abs(b) < 1e-6:
        return 1.0
    return a / b

OPERATORS['/'] = safe_divide

def generate_tree(max_depth, method='grow', depth=0):
    """Gera uma árvore de expressão aleatória"""
    if depth >= max_depth - 1 or (method == 'grow' and random.random() < 0.5):
        # Gera terminal
        if random.random() < 0.5:
            return Node('x')
        else:
            return Node(f'c{random.randint(0,9)}')
    else:
        # Gera operador
        op = random.choice(list(OPERATORS.keys()))
        if op in ['sin', 'cos']:  # Operadores unários
            return Node(op, left=generate_tree(max_depth, method, depth+1))
        else:  # Operadores binários
            return Node(
                op,
                left=generate_tree(max_depth, method, depth+1),
                right=generate_tree(max_depth, method, depth+1)
            )

def tree_to_expression(node):
    """Converte a árvore em expressão matemática"""
    if node is None:
        return ''
        
    if node.left is None and node.right is None:
        return node.value
    elif node.right is None:  # Operador unário
        return f"{node.value}({tree_to_expression(node.left)})"
    else:
        return f"({tree_to_expression(node.left)} {node.value} {tree_to_expression(node.right)})"

# ================================
# FUNÇÃO DE FITNESS
# ================================
def evaluate_tree(node, x, constants):
    """Avalia a árvore para um dado valor de x"""
    if node is None:
        return 0.0
    
    try:
        if node.value == 'x':
            return x
        elif node.value.startswith('c'):
            return constants[int(node.value[1])]
        elif node.value in ['sin', 'cos']:
            arg = evaluate_tree(node.left, x, constants)
            return OPERATORS[node.value](arg)
        else:
            left_val = evaluate_tree(node.left, x, constants)
            right_val = evaluate_tree(node.right, x, constants)
            return OPERATORS[node.value](left_val, right_val)
    except (ValueError, TypeError, OverflowError):
        return 0.0

def fitness(individual, X, y, constants):
    """Calcula o erro quadrático médio (MSE)"""
    errors = []
    for i, x_val in enumerate(X):
        try:
            pred = evaluate_tree(individual, x_val, constants)
            errors.append((pred - y[i])**2)
        except Exception:
            errors.append(1e10)  # Penalidade alta para expressões inválidas
            
    return np.mean(errors) if errors else 1e10

# ================================
# OPERADORES GENÉTICOS
# ================================
def select_parents(population, fitnesses, tournament_size=3):
    """Seleção por torneio"""
    selected = []
    for _ in range(2):
        contestants = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(contestants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    """Cruzamento por troca de subárvores"""
    if parent1 is None or parent2 is None:
        return parent1, parent2
        
    def find_crossover_point(node, depth=0):
        if node is None:
            return []
        points = [(node, depth)]
        if node.left:
            points.extend(find_crossover_point(node.left, depth+1))
        if node.right:
            points.extend(find_crossover_point(node.right, depth+1))
        return points
    
    points1 = find_crossover_point(parent1)
    points2 = find_crossover_point(parent2)
    
    if not points1 or not points2:
        return parent1, parent2
    
    node1, depth1 = random.choice(points1)
    node2, depth2 = random.choice(points2)
    
    # Troca as subárvores
    node1.left, node2.left = node2.left, node1.left
    node1.right, node2.right = node2.right, node1.right
    
    return parent1, parent2

def mutate(individual, max_depth=4):
    """Mutação por substituição de subárvore"""
    if individual is None:
        return generate_tree(max_depth)
        
    def find_mutation_point(node, depth=0):
        if node is None:
            return []
        points = [(node, depth)]
        if node.left:
            points.extend(find_mutation_point(node.left, depth+1))
        if node.right:
            points.extend(find_mutation_point(node.right, depth+1))
        return points
    
    points = find_mutation_point(individual)
    if not points:
        return individual
    
    node, depth = random.choice(points)
    new_subtree = generate_tree(max(1, max_depth - depth), method='grow')
    
    # Substitui a subárvore
    if node.left is not None:
        node.left = new_subtree
    elif node.right is not None:
        node.right = new_subtree
    else:
        # Se for um nó folha, substitui o valor
        node.value = new_subtree.value
        node.left = new_subtree.left
        node.right = new_subtree.right
    
    return individual

# ================================
# ALGORITMO GENÉTICO TRADICIONAL
# ================================
def genetic_algorithm(X, y, pop_size=100, max_depth=4, generations=50, 
                      cx_prob=0.7, mut_prob=0.2):
    """Executa o AG tradicional com uma única população"""
    # Gerar população inicial
    population = [generate_tree(max_depth) for _ in range(pop_size)]
    constants = [random.uniform(-10, 10) for _ in range(10)]
    
    best_fitness_history = []
    avg_fitness_history = []
    
    for gen in range(generations):
        # Avaliar fitness
        fitnesses = [fitness(ind, X, y, constants) for ind in population]
        
        # Registrar estatísticas
        best_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Selecionar nova população
        new_population = []
        while len(new_population) < pop_size:
            parents = select_parents(population, fitnesses)
            offspring1, offspring2 = deepcopy(parents[0]), deepcopy(parents[1])
            
            # Cruzamento
            if random.random() < cx_prob:
                try:
                    offspring1, offspring2 = crossover(offspring1, offspring2)
                except:
                    pass  # Mantém os indivíduos originais em caso de erro
            
            # Mutação
            if random.random() < mut_prob:
                offspring1 = mutate(offspring1, max_depth)
            if random.random() < mut_prob:
                offspring2 = mutate(offspring2, max_depth)
            
            new_population.extend([offspring1, offspring2])
        
        population = new_population[:pop_size]
    
    # Encontrar melhor indivíduo
    fitnesses = [fitness(ind, X, y, constants) for ind in population]
    best_idx = np.argmin(fitnesses)
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    return best_individual, best_fitness, best_fitness_history, avg_fitness_history

# ================================
# ALGORITMO GENÉTICO COM ILHAS
# ================================
def island_genetic_algorithm(X, y, num_islands=4, pop_per_island=50, 
                             max_depth=4, generations=100, cx_prob=0.7, 
                             mut_prob=0.2, migration_interval=10, 
                             migration_rate=0.1):
    """Executa o AG com estratégia de ilhas"""
    # Inicializar ilhas
    islands = []
    constants = [[random.uniform(-10, 10) for _ in range(10)] for _ in range(num_islands)]
    
    for _ in range(num_islands):
        island_pop = [generate_tree(max_depth) for _ in range(pop_per_island)]
        islands.append(island_pop)
    
    best_fitness_history = [[] for _ in range(num_islands)]
    avg_fitness_history = [[] for _ in range(num_islands)]
    
    for gen in range(generations):
        # Processar cada ilha
        for i in range(num_islands):
            # Avaliar fitness
            fitnesses = [fitness(ind, X, y, constants[i]) for ind in islands[i]]
            
            # Registrar estatísticas
            best_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness_history[i].append(best_fitness)
            avg_fitness_history[i].append(avg_fitness)
            
            # Evolução
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
        
        # Migração entre ilhas (topologia em anel)
        if gen % migration_interval == 0 and gen > 0:
            for i in range(num_islands):
                next_island = (i + 1) % num_islands
                
                # Selecionar emigrantes
                num_emigrants = int(migration_rate * pop_per_island)
                emigrants = random.sample(islands[i], num_emigrants)
                
                # Substituir imigrantes
                for emigrant in emigrants:
                    idx = random.randint(0, pop_per_island - 1)
                    islands[next_island][idx] = deepcopy(emigrant)
    
    # Encontrar melhor indivíduo global
    best_global = None
    best_global_fitness = float('inf')
    for i in range(num_islands):
        fitnesses = [fitness(ind, X, y, constants[i]) for ind in islands[i]]
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_global_fitness:
            best_global = islands[i][best_idx]
            best_global_fitness = fitnesses[best_idx]
    
    return best_global, best_global_fitness, best_fitness_history, avg_fitness_history

# ================================
# TESTES E COMPARAÇÃO
# ================================
def generate_target_function():
    """Gera a função alvo e dados de treino"""
    def target_func(x):
        return x**3 - 2*x**2 + math.sin(x)
    
    X_train = np.linspace(-5, 5, 50)
    y_train = np.array([target_func(x) for x in X_train])
    
    return X_train, y_train, target_func

def run_comparison():
    """Executa e compara ambos os algoritmos"""
    X, y, target_func = generate_target_function()
    
    # Executar AG tradicional
    print("Executando AG tradicional...")
    start_time = time.time()
    best_individual_simple, best_fitness_simple, best_hist_simple, avg_hist_simple = genetic_algorithm(
        X, y, pop_size=100, generations=100
    )
    simple_time = time.time() - start_time
    
    # Executar AG com ilhas
    print("Executando AG com ilhas...")
    start_time = time.time()
    best_individual_island, best_fitness_island, best_hist_island, avg_hist_island = island_genetic_algorithm(
        X, y, num_islands=4, pop_per_island=50, generations=100
    )
    island_time = time.time() - start_time
    
    # Resultados
    print("\n" + "="*50)
    print("COMPARAÇÃO DOS ALGORITMOS")
    print("="*50)
    print(f"AG Tradicional:")
    print(f"  Melhor fitness: {best_fitness_simple:.6f}")
    print(f"  Expressão: {tree_to_expression(best_individual_simple)}")
    print(f"  Tempo de execução: {simple_time:.2f} segundos")
    
    print(f"\nAG com Ilhas:")
    print(f"  Melhor fitness: {best_fitness_island:.6f}")
    print(f"  Expressão: {tree_to_expression(best_individual_island)}")
    print(f"  Tempo de execução: {island_time:.2f} segundos")
    
    # Plotar convergência
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(best_hist_simple, 'b-', label='Melhor Fitness')
    plt.plot(avg_hist_simple, 'r-', label='Fitness Médio')
    plt.title('AG Tradicional - Convergência')
    plt.xlabel('Geração')
    plt.ylabel('Fitness (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i, hist in enumerate(best_hist_island):
        plt.plot(hist, label=f'Ilha {i+1}')
    plt.title('AG com Ilhas - Melhor Fitness por Ilha')
    plt.xlabel('Geração')
    plt.ylabel('Fitness (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png')
    plt.show()
    
    # Plotar soluções
    plt.figure(figsize=(10, 6))
    X_test = np.linspace(-5, 5, 100)
    y_target = [target_func(x) for x in X_test]
    
    constants_simple = [random.uniform(-10, 10) for _ in range(10)]
    constants_island = [random.uniform(-10, 10) for _ in range(10)]
    
    y_pred_simple = [evaluate_tree(best_individual_simple, x, constants_simple) for x in X_test]
    y_pred_island = [evaluate_tree(best_individual_island, x, constants_island) for x in X_test]
    
    plt.plot(X_test, y_target, 'k-', linewidth=2, label='Função Alvo')
    plt.plot(X_test, y_pred_simple, 'b--', label='AG Tradicional')
    plt.plot(X_test, y_pred_island, 'r-.', label='AG com Ilhas')
    
    plt.title('Comparação das Soluções Encontradas')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('solutions_comparison.png')
    plt.show()

# Executar a comparação
if __name__ == "__main__":
    run_comparison()