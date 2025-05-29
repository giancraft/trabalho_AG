import math
import numpy as np
from representation import OPERATORS, Node

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

def evaluate_tree(node, x, constants):
    if node is None:
        return 0.0
    
    try:
        if node.value == 'x':
            return x
        elif node.value.startswith('c'):
            return constants[int(node.value[1])]
        elif node.value in ['sin', 'cos']:
            arg = evaluate_tree(node.left, x, constants)
            return OPERATORS_FUNC[node.value](arg)
        else:
            left_val = evaluate_tree(node.left, x, constants)
            right_val = evaluate_tree(node.right, x, constants)
            return OPERATORS_FUNC[node.value](left_val, right_val)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return 0.0

def fitness(individual, X, y, constants):
    errors = []
    for i, x_val in enumerate(X):
        try:
            pred = evaluate_tree(individual, x_val, constants)
            errors.append((pred - y[i])**2)
        except Exception:
            errors.append(1e10)
            
    return np.mean(errors) if errors else 1e10