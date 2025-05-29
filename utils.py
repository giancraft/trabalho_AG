import math
import numpy as np

def generate_target_function():
    def target_func(x):
        return x**3 - 2*x**2 + math.sin(x)
    
    X_train = np.linspace(-5, 5, 50)
    y_train = np.array([target_func(x) for x in X_train])
    
    return X_train, y_train, target_func