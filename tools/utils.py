import numpy as np
import torch

def init_random_seed(seed=None):
    if seed is None:
        seed = np.random.randint(2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
