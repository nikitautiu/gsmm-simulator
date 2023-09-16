import random
import numpy as np

global_seed = None

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    global_seed = seed

def get_current_seed():
    return global_seed
