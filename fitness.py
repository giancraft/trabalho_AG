import math
import random
import numpy as np
from representation import OPERATORS, Node
from scipy.optimize import least_squares

def safe_divide(a, b):
    if abs(b) < 1e-6:
        return 1.0
    return a / b

OPERATORS_FUNC = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': safe_divide,
    'sin': math.sin,
    'cos': math.cos
}

def evaluate_tree(node, x, constants, depth=0, max_depth=50):
    if depth > max_depth:
        return 0.0  # Prevenir recursão infinita
    
    if node is None:
        return 0.0
    
    try:
        if node.value == 'x':
            return x
        elif node.value.startswith('c'):
            return constants[int(node.value[1])]
        elif node.value in ['sin', 'cos']:
            arg = evaluate_tree(node.left, x, constants, depth+1, max_depth)
            return OPERATORS_FUNC[node.value](arg)
        else:
            left_val = evaluate_tree(node.left, x, constants, depth+1, max_depth)
            right_val = evaluate_tree(node.right, x, constants, depth+1, max_depth)
            return OPERATORS_FUNC[node.value](left_val, right_val)
    except (ValueError, TypeError, OverflowError):
        return 0.0

def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def fitness(individual, X, y):
    tree, constants = individual
    errors = []
    for i, x_val in enumerate(X):
        try:
            pred = evaluate_tree(tree, x_val, constants)
            errors.append((pred - y[i])**2)
        except Exception:
            errors.append(1e10)
            
    return np.mean(errors) if errors else 1e10

def optimize_constants(tree, X, y, initial_constants):
    """Otimização simplificada de constantes usando amostragem estocástica"""
    best_constants = initial_constants
    best_error = 1e10
    
    # Avaliar erro inicial
    errors = []
    for i, x_val in enumerate(X):
        try:
            pred = evaluate_tree(tree, x_val, best_constants)
            errors.append((pred - y[i])**2)
        except Exception:
            errors.append(1e10)
    best_error = np.mean(errors) if errors else 1e10
    
    # Tentativas de otimização
    for _ in range(20):  # Número limitado de tentativas
        new_constants = [
            c + random.gauss(0, 0.5)  # Pequena perturbação
            for c in best_constants
        ]
        
        errors = []
        for i, x_val in enumerate(X):
            try:
                pred = evaluate_tree(tree, x_val, new_constants)
                errors.append((pred - y[i])**2)
            except Exception:
                errors.append(1e10)
        
        new_error = np.mean(errors) if errors else 1e10
        
        if new_error < best_error:
            best_constants = new_constants
            best_error = new_error
    
    return best_constants