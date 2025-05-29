import random
from copy import deepcopy
from representation import generate_tree

def select_parents(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(2):
        contestants = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(contestants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
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
    
    node1.left, node2.left = node2.left, node1.left
    node1.right, node2.right = node2.right, node1.right
    
    return parent1, parent2

def mutate(individual, max_depth=4):
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
    
    if node.left is not None:
        node.left = new_subtree
    elif node.right is not None:
        node.right = new_subtree
    else:
        node.value = new_subtree.value
        node.left = new_subtree.left
        node.right = new_subtree.right
    
    return individual

def mutate_constants(individual, mutation_rate=0.1):
    tree, constants = individual
    new_constants = [
        c + random.gauss(0, 0.5) if random.random() < mutation_rate else c
        for c in constants
    ]
    return tree, new_constants