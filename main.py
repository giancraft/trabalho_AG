import random
import time
import matplotlib.pyplot as plt
import numpy as np
from representation import tree_to_expression
from fitness import evaluate_tree
from simple_ga import genetic_algorithm
from island_ga import island_genetic_algorithm
from utils import generate_target_function

def run_comparison():
    X, y, target_func = generate_target_function()
    
    # AG tradicional
    print("Executando AG tradicional...")
    start_time = time.time()
    # Note: genetic_algorithm agora retorna (tree, constants) no best_individual
    result_simple = genetic_algorithm(X, y, pop_size=200, generations=100)
    simple_time = time.time() - start_time
    # Desempacotar resultado
    best_individual_simple, best_fitness_simple, best_hist_simple, avg_hist_simple = result_simple[:4]
    
    # AG com ilhas
    print("Executando AG com ilhas...")
    start_time = time.time()
    result_island = island_genetic_algorithm(
        X, y, num_islands=4, pop_per_island=50, generations=100
    )
    island_time = time.time() - start_time
    best_individual_island, best_fitness_island, best_hist_island, avg_hist_island = result_island[:4]
    
    # Resultados
    print("\n" + "="*50)
    print("COMPARAÇÃO DOS ALGORITMOS")
    print("="*50)
    print(f"AG Tradicional:")
    print(f"  Melhor fitness: {best_fitness_simple:.6f}")
    # O indivíduo agora é (tree, constants) -> extrair a árvore para expressão
    print(f"  Expressão: {tree_to_expression(best_individual_simple[0])}")
    print(f"  Tempo de execução: {simple_time:.2f} segundos")
    
    print(f"\nAG com Ilhas:")
    print(f"  Melhor fitness: {best_fitness_island:.6f}")
    print(f"  Expressão: {tree_to_expression(best_individual_island[0])}")
    print(f"  Tempo de execução: {island_time:.2f} segundos")
    
    # Gráficos de convergência
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
    
    # Comparação de soluções
    plt.figure(figsize=(10, 6))
    X_test = np.linspace(-5, 5, 100)
    y_target = [target_func(x) for x in X_test]
    
    # Para AG tradicional: best_individual_simple é (tree, constants)
    tree_simple, constants_simple = best_individual_simple
    y_pred_simple = [evaluate_tree(tree_simple, x, constants_simple) for x in X_test]
    
    # Para AG com ilhas: best_individual_island é (tree, constants)
    tree_island, constants_island = best_individual_island
    y_pred_island = [evaluate_tree(tree_island, x, constants_island) for x in X_test]
    
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

if __name__ == "__main__":
    run_comparison()