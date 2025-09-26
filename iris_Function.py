import numpy as np

def hamming_distance(c1, c2):
    return np.sum(c1 != c2) / len(c1)
