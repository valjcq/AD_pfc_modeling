import numpy as np


def k_input_output(z, alpha, theta):
    return np.exp(alpha * theta / (1 + z)) / (1 + np.exp(alpha * theta / (1 + z)))
