import random

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
    def __str__(self):
        if self.left is None and self.right is None:
            return self.value
        elif self.right is None:
            return f"{self.value}({self.left})"
        else:
            return f"({self.left} {self.value} {self.right})"

# Operadores e terminais
OPERATORS = {
    '+': 'binary',
    '-': 'binary',
    '*': 'binary',
    '/': 'binary',
    'sin': 'unary',
    'cos': 'unary'
}

TERMINALS = ['x'] + [f'c{i}' for i in range(10)]

def generate_tree(max_depth, method='grow', depth=0):
    if depth >= max_depth - 1 or (method == 'grow' and random.random() < 0.5):
        if random.random() < 0.5:
            return Node('x')
        else:
            return Node(f'c{random.randint(0,9)}')
    else:
        op = random.choice(list(OPERATORS.keys()))
        op_type = OPERATORS[op]
        
        if op_type == 'unary':
            return Node(op, left=generate_tree(max_depth, method, depth+1))
        else:
            return Node(
                op,
                left=generate_tree(max_depth, method, depth+1),
                right=generate_tree(max_depth, method, depth+1)
            )

def tree_to_expression(node):
    if node is None:
        return ''
        
    if node.left is None and node.right is None:
        return node.value
    elif node.right is None:
        return f"{node.value}({tree_to_expression(node.left)})"
    else:
        return f"({tree_to_expression(node.left)} {node.value} {tree_to_expression(node.right)})"
    
def generate_individual(max_depth):
    tree = generate_tree(max_depth)
    constants = [random.uniform(-10, 10) for _ in range(10)]
    return tree, constants

def count_nodes(node):
    """Conta o número de nós na árvore"""
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def simplify_tree(node):
    if node is None:
        return None
        
    # Simplificar sub-árvores
    node.left = simplify_tree(node.left)
    node.right = simplify_tree(node.right)
    
    # Apenas aplicar regras se ambos os filhos existirem
    if node.left and node.right:
        if node.value == '*':
            if node.left.value == '0' or node.right.value == '0':
                return Node('0')
            if node.left.value == '1':
                return node.right
            if node.right.value == '1':
                return node.left
                
        elif node.value == '+':
            if node.left.value == '0':
                return node.right
            if node.right.value == '0':
                return node.left
                
        elif node.value == '-':
            if node.right.value == '0':
                return node.left
    
    return node