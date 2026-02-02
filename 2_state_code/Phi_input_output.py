import numpy as np


def Phi_input_output(x, y, z, alpha, theta):
    return 1 / (1 + np.exp(-alpha / (1 + z) * (x - (theta + y)))) - 1 / (
        1 + np.exp(alpha * theta / (1 + z))
    )
