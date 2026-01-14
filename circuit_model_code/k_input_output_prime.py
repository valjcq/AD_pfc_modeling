import numpy as np


def k_input_output_prime_z(z, alpha, theta):
    return (
        alpha
        * theta
        / ((1 + z) ** 2)
        * np.exp(alpha * theta / (1 + z))
        / (1 + np.exp(alpha * theta / (1 + z)))
        * (-1 + np.exp(alpha * theta / (1 + z)) / (1 + np.exp(alpha * theta / (1 + z))))
    )
