import numpy as np 
           
POP_SIZE = 50           
CROSS_RATE = 0.8         
MUTATION_RATE = 0.003    
N_GENERATIONS = 30

"""
the code references from https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Genetic%20Algorithm/Genetic%20Algorithm%20Basic.py
"""
def select(pop,fitness):
    idx =  np.random.choice(np.arange(POP_SIZE),size = POP_SIZE,replace=True, p = fitness/np.sum(fitness))
    return pop[idx]

def crossover(parent, pop,DNA_SIZE):     
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  
        parent[cross_points] = pop[i_, cross_points]                            
    return parent


def mutate(child,DNA_SIZE):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


