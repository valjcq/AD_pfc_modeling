import numpy as np


def Phi_input_output_prime_x(x, y, z, alpha, theta):
    return (
        alpha
        / (1 + z)
        * np.exp(-alpha / (1 + z) * (x - (theta + y)))
        / ((1 + np.exp(-alpha / (1 + z) * (x - (theta + y)))) ** 2)
    )


def Phi_input_output_prime_y(x, y, z, alpha, theta):
    return (
        -alpha
        / (1 + z)
        * np.exp(-alpha / (1 + z) * (x - (theta + y)))
        / ((1 + np.exp(-alpha / (1 + z) * (x - (theta + y)))) ** 2)
    )


def Phi_input_output_prime_z(x, y, z, alpha, theta):
    return -alpha * (x - (theta + y)) / ((1 + z) ** 2) * np.exp(
        -alpha / (1 + z) * (x - (theta + y))
    ) / ((1 + np.exp(-alpha / (1 + z) * (x - (theta + y)))) ** 2) - alpha * theta * (
        (1 + z) ** 2
    ) * np.exp(
        alpha * theta / (1 + z)
    ) / (
        (1 + np.exp(alpha * theta / (1 + z))) ** 2
    )


def Phi_input_output_prime_xyz(x, y, z, xprime, yprime, zprime, alpha, theta):
    return alpha * ((xprime - yprime) * (1 + z) - zprime * (x - (theta + y))) / (
        (1 + z) ** 2
    ) * np.exp(-alpha / (1 + z) * (x - (theta + y))) / (
        (1 + np.exp(-alpha / (1 + z) * (x - (theta + y)))) ** 2
    ) - alpha * theta * zprime / (
        (1 + z) ** 2
    ) * np.exp(
        alpha * theta / (1 + z)
    ) / (
        (1 + np.exp(alpha * theta / (1 + z))) ** 2
    )


def Phi_input_output_prime_r_som_r_pv_x(
    x, r_som, r_pv, r_som_prime, r_pv_prime, w_11, w_21, w_31, alpha, theta
):
    return alpha * (
        (w_11 - w_31 * r_som_prime) * (1 + w_21 * r_pv)
        - w_21 * r_pv_prime * (w_11 * x - w_31 * r_som - theta)
    ) / ((1 + w_21 * r_pv) ** 2) * np.exp(
        -alpha / (1 + w_21 * r_pv) * (w_11 * x - (theta + w_31 * r_som))
    ) / (
        (1 + np.exp(-alpha / (1 + w_21 * r_pv) * (w_11 * x - (theta + w_31 * r_som))))
        ** 2
    ) - alpha * theta * w_21 * r_pv_prime / (
        (1 + w_21 * r_pv) ** 2
    ) * np.exp(
        alpha * theta / (1 + w_21 * r_pv)
    ) / (
        (1 + np.exp(alpha * theta / (1 + w_21 * r_pv))) ** 2
    )


def Phi_input_output_prime_r_som_r_pv_y(
    x, r_som, r_pv, r_pv_prime, w_11, w_21, w_31, alpha, theta
):
    return -alpha * w_21 * r_pv_prime * (w_11 * x - w_31 * r_som - theta) / (
        (1 + w_21 * r_pv) ** 2
    ) * np.exp(-alpha / (1 + w_21 * r_pv) * (w_11 * x - (theta + w_31 * r_som))) / (
        (1 + np.exp(-alpha / (1 + w_21 * r_pv) * (w_11 * x - (theta + w_31 * r_som))))
        ** 2
    ) - alpha * theta * w_21 * r_pv_prime / (
        (1 + w_21 * r_pv) ** 2
    ) * np.exp(
        alpha * theta / (1 + w_21 * r_pv)
    ) / (
        (1 + np.exp(alpha * theta / (1 + w_21 * r_pv))) ** 2
    )
