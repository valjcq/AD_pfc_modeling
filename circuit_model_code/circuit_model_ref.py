import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

from itertools import product
from Phi_input_output import Phi_input_output
from k_input_output import k_input_output
from Phi_input_output_prime import (
    Phi_input_output_prime_x,
    Phi_input_output_prime_y,
    Phi_input_output_prime_xyz,
)
from k_input_output_prime import k_input_output_prime_z
from Phi_heav import Phi_heav
from math import sqrt
from scipy.optimize import fsolve
from numpy import linalg as LA

from original_params import (
    w_11,
    w_21,
    w_31,
    w_12,
    w_32,
    w_42,
    w_22,
    w_13,
    w_23,
    w_43,
    w_33,
    w_14,
    w_24,
    w_34,
    w_44,
    i_pyr,
    i_inter_pv_max,
    i_inter_som_max,
    i_inter_vip_max,
    J_r1,
    alpha_pyr,
    theta_pyr,
    alpha_inter_pv,
    theta_inter_pv,
    alpha_inter_som,
    theta_inter_som,
    alpha_inter_vip,
    theta_inter_vip,
    kd,
    fact_pyr,
    fact_pv,
    fact_som,
    fact_vip,
    tau_a_1,
    sigma,
    mult_factor_i,
)

SEED_VALUE = 42
np.random.seed(SEED_VALUE)

tmax = 50000  # duration of simulations, in seconds.
dt = 0.001  # time step, in seconds.
tau_s = 0.02  # integration time, in seconds.

i_inter_pv = i_inter_pv_max * mult_factor_i
i_inter_som = i_inter_som_max * mult_factor_i
i_inter_vip = i_inter_vip_max * mult_factor_i


def equations(
    x,
    w_11: float,
    w_21: float,
    w_31: float,
    w_12: float,
    w_32: float,
    w_42: float,
    w_22: float,
    w_13: float,
    w_23: float,
    w_43: float,
    w_33: float,
    w_14: float,
    w_24: float,
    w_34: float,
    w_44: float,
    i_pyr: float,
    i_inter_pv: float,
    i_inter_som: float,
    i_inter_vip: float,
    J_r1: float,
):
    """
    Defines the system of differential equations governing the firing-rate dynamics
    of four interacting neural populations:
        1. Pyramidal cells (PYR)
        2. Parvalbumin-positive interneurons (PV)
        3. Somatostatin-positive interneurons (SOM)
        4. VIP-positive interneurons (VIP)

    Parameters
    ----------
    x : array-like of shape (4,)
        Current firing rates of the four populations: [PYR, PV, SOM, VIP].
    w_ij : float
        Synaptic connection weight from population j to population i.
        (e.g., w_21 is the weight from PV → PYR).
    i_pyr, i_inter_pv, i_inter_som, i_inter_vip : float
        External input currents to each population.
    J_r1 : float
        Recurrent self-excitation term specific to the pyramidal population.

    Returns
    -------
    list of float
        The time derivatives (d/dt) for each population's activity:
        [dPYR/dt, dPV/dt, dSOM/dt, dVIP/dt].
    """

    # --- PYRAMIDAL CELLS (excitatory population) ---
    f = (
        k_input_output(kd * w_21 * x[1], alpha_pyr, theta_pyr) - x[0]
    ) * Phi_input_output(
        # Excitatory and inhibitory synaptic inputs to PYR
        w_11 * x[0] + i_pyr,  # recurrent + external excitation
        (1 - kd) * w_21 * x[1]
        + w_31 * x[2]
        + J_r1 * x[0],  # inhibitory & modulatory terms
        kd * w_21 * x[1],  # scaling term for inhibition
        alpha_pyr,
        theta_pyr,
    ) - x[0]

    # --- PARVALBUMIN (fast-spiking inhibitory interneurons) ---
    g = (k_input_output(0, alpha_inter_pv, theta_inter_pv) - x[1]) * Phi_input_output(
        w_12 * x[0] + i_inter_pv,  # excitatory drive from PYR + external input
        w_32 * x[2] + w_22 * x[1] + w_42 * x[3],  # inhibition from SOM, PV, VIP
        0,
        alpha_inter_pv,
        theta_inter_pv,
    ) - x[1]

    # --- SOMATOSTATIN (slow inhibitory interneurons) ---
    h = (k_input_output(0, alpha_inter_som, theta_inter_som) - x[2]) * Phi_input_output(
        w_13 * x[0] + i_inter_som,  # excitation from PYR + external input
        w_23 * x[1] + w_33 * x[2] + w_43 * x[3],  # inhibition from PV, SOM, VIP
        0,
        alpha_inter_som,
        theta_inter_som,
    ) - x[2]

    # --- VIP (disinhibitory interneurons) ---
    i = (k_input_output(0, alpha_inter_vip, theta_inter_vip) - x[3]) * Phi_input_output(
        w_14 * x[0] + i_inter_vip,  # excitation from PYR + external input
        w_24 * x[1] + w_44 * x[3] + w_34 * x[2],  # inhibition from PV, VIP, SOM
        0,
        alpha_inter_vip,
        theta_inter_vip,
    ) - x[3]

    # Return the derivatives for all four populations
    return [f, g, h, i]


def equations_prime(
    x,
    w_11: float,
    w_21: float,
    w_31: float,
    w_12: float,
    w_32: float,
    w_42: float,
    w_22: float,
    w_13: float,
    w_23: float,
    w_43: float,
    w_33: float,
    w_14: float,
    w_24: float,
    w_34: float,
    w_44: float,
    i_pyr: float,
    i_inter_pv: float,
    i_inter_som: float,
    i_inter_vip: float,
    J_r1: float,
):
    """
    Computes the Jacobian matrix of the firing-rate dynamics system
    for four interacting neural populations (PYR, PV, SOM, VIP).

    Parameters
    ----------
    x : array-like of shape (4,)
        Current firing rates: [PYR, PV, SOM, VIP].
    w_ij : float
        Synaptic connection weights from population j to i.
    i_pyr, i_inter_* : float
        External input currents to each population.
    J_r1 : float
        Recurrent self-excitation term for PYR.

    Returns
    -------
    list of lists
        4x4 Jacobian matrix where entry (i,j) is the derivative
        of the i-th population's dynamics with respect to the j-th population.
    """

    # --- PYRAMIDAL CELLS (PYR) ---
    # Partial derivatives w.r.t. PYR, PV, SOM, VIP
    a = (
        -Phi_input_output(
            w_11 * x[0] + i_pyr,
            (1 - kd) * w_21 * x[1] + w_31 * x[2] + J_r1 * x[0],
            kd * w_21 * x[1],
            alpha_pyr,
            theta_pyr,
        )
        + (k_input_output(kd * w_21 * x[1], alpha_pyr, theta_pyr) - x[0])
        * (w_11 - J_r1)
        * Phi_input_output_prime_x(
            w_11 * x[0] + i_pyr,
            (1 - kd) * w_21 * x[1] + w_31 * x[2] + J_r1 * x[0],
            kd * w_21 * x[1],
            alpha_pyr,
            theta_pyr,
        )
        - 1
    )
    b = kd * w_21 * k_input_output_prime_z(
        kd * w_21 * x[1], alpha_pyr, theta_pyr
    ) * Phi_input_output(
        w_11 * x[0] + i_pyr,
        (1 - kd) * w_21 * x[1] + w_31 * x[2] + J_r1 * x[0],
        kd * w_21 * x[1],
        alpha_pyr,
        theta_pyr,
    ) + (
        k_input_output(kd * w_21 * x[1], alpha_pyr, theta_pyr) - x[0]
    ) * Phi_input_output_prime_xyz(
        w_11 * x[0] + i_pyr,
        (1 - kd) * w_21 * x[1] + w_31 * x[2] + J_r1 * x[0],
        kd * w_21 * x[1],
        0,
        (1 - kd) * w_21,
        kd * w_21,
        alpha_pyr,
        theta_pyr,
    )
    c = (
        (k_input_output(kd * w_21 * x[1], alpha_pyr, theta_pyr) - x[0])
        * w_31
        * Phi_input_output_prime_y(
            w_11 * x[0] + i_pyr,
            (1 - kd) * w_21 * x[1] + w_31 * x[2] + J_r1 * x[0],
            kd * w_21 * x[1],
            alpha_pyr,
            theta_pyr,
        )
    )
    d = 0  # derivative w.r.t VIP is zero

    # --- PARVALBUMIN (PV) ---
    e = (
        (k_input_output(0, alpha_inter_pv, theta_inter_pv) - x[1])
        * w_12
        * Phi_input_output_prime_x(
            w_12 * x[0] + i_inter_pv,
            w_32 * x[2] + w_22 * x[1] + w_42 * x[3],
            0,
            alpha_inter_pv,
            theta_inter_pv,
        )
    )
    f = (
        -Phi_input_output(
            w_12 * x[0] + i_inter_pv,
            w_32 * x[2] + w_22 * x[1] + w_42 * x[3],
            0,
            alpha_inter_pv,
            theta_inter_pv,
        )
        + (k_input_output(0, alpha_inter_pv, theta_inter_pv) - x[1])
        * w_22
        * Phi_input_output_prime_y(
            w_12 * x[0] + i_inter_pv,
            w_32 * x[2] + w_22 * x[1] + w_42 * x[3],
            0,
            alpha_inter_pv,
            theta_inter_pv,
        )
        - 1
    )
    g = (
        (k_input_output(0, alpha_inter_pv, theta_inter_pv) - x[1])
        * w_32
        * Phi_input_output_prime_y(
            w_12 * x[0] + i_inter_pv,
            w_32 * x[2] + w_22 * x[1] + w_42 * x[3],
            0,
            alpha_inter_pv,
            theta_inter_pv,
        )
    )
    h = (
        (k_input_output(0, alpha_inter_pv, theta_inter_pv) - x[1])
        * w_42
        * Phi_input_output_prime_y(
            w_12 * x[0] + i_inter_pv,
            w_32 * x[2] + w_22 * x[1] + w_42 * x[3],
            0,
            alpha_inter_pv,
            theta_inter_pv,
        )
    )

    # --- SOMATOSTATIN (SOM) ---
    i = (
        (k_input_output(0, alpha_inter_som, theta_inter_som) - x[2])
        * w_13
        * Phi_input_output_prime_x(
            w_13 * x[0] + i_inter_som,
            w_23 * x[1] + w_33 * x[2] + w_43 * x[3],
            0,
            alpha_inter_som,
            theta_inter_som,
        )
    )
    j = (
        (k_input_output(0, alpha_inter_som, theta_inter_som) - x[2])
        * w_23
        * Phi_input_output_prime_y(
            w_13 * x[0] + i_inter_som,
            w_23 * x[1] + w_33 * x[2] + w_43 * x[3],
            0,
            alpha_inter_som,
            theta_inter_som,
        )
    )
    k = (
        -Phi_input_output(
            w_13 * x[0] + i_inter_som,
            w_23 * x[1] + w_33 * x[2] + w_43 * x[3],
            0,
            alpha_inter_som,
            theta_inter_som,
        )
        + (k_input_output(0, alpha_inter_som, theta_inter_som) - x[2])
        * w_33
        * Phi_input_output_prime_y(
            w_13 * x[0] + i_inter_som,
            w_23 * x[1] + w_33 * x[2] + w_43 * x[3],
            0,
            alpha_inter_som,
            theta_inter_som,
        )
        - 1
    )
    l = (
        (k_input_output(0, alpha_inter_som, theta_inter_som) - x[2])
        * w_43
        * Phi_input_output_prime_y(
            w_13 * x[0] + i_inter_som,
            w_23 * x[1] + w_33 * x[2] + w_43 * x[3],
            0,
            alpha_inter_som,
            theta_inter_som,
        )
    )

    # --- VIP ---
    m = (
        (k_input_output(0, alpha_inter_vip, theta_inter_vip) - x[3])
        * w_14
        * Phi_input_output_prime_x(
            w_14 * x[0] + i_inter_vip,
            w_24 * x[1] + w_44 * x[3] + w_34 * x[2],
            0,
            alpha_inter_vip,
            theta_inter_vip,
        )
    )
    n = (
        (k_input_output(0, alpha_inter_vip, theta_inter_vip) - x[3])
        * w_24
        * Phi_input_output_prime_y(
            w_14 * x[0] + i_inter_vip,
            w_24 * x[1] + w_44 * x[3] + w_34 * x[2],
            0,
            alpha_inter_vip,
            theta_inter_vip,
        )
    )
    o = (
        (k_input_output(0, alpha_inter_vip, theta_inter_vip) - x[3])
        * w_34
        * Phi_input_output_prime_y(
            w_14 * x[0] + i_inter_vip,
            w_24 * x[1] + w_44 * x[3] + w_34 * x[2],
            0,
            alpha_inter_vip,
            theta_inter_vip,
        )
    )
    p = (
        -Phi_input_output(
            w_14 * x[0] + i_inter_vip,
            w_24 * x[1] + w_44 * x[3] + w_34 * x[2],
            0,
            alpha_inter_vip,
            theta_inter_vip,
        )
        + (k_input_output(0, alpha_inter_vip, theta_inter_vip) - x[3])
        * w_44
        * Phi_input_output_prime_y(
            w_14 * x[0] + i_inter_vip,
            w_24 * x[1] + w_44 * x[3] + w_34 * x[2],
            0,
            alpha_inter_vip,
            theta_inter_vip,
        )
        - 1
    )

    # Return the 4x4 Jacobian matrix
    return [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]


def find_steady_states():
    """
    Find and classify the steady states of a 4-population firing-rate system.

    The function:
        1. Generates a grid of initial conditions for all populations.
        2. Solves for steady states using `fsolve`.
        3. Removes duplicate solutions.
        4. Computes the Jacobian at each steady state and calculates eigenvalues.
        5. Classifies steady states as stable (all eigenvalues <= 0) or unstable (any eigenvalue > 0).
        6. Sorts steady states by the first population (PYR) activity.

    Returns
    -------
    stable_paramSolve : np.ndarray
        Array of stable steady states, each row corresponding to a solution
        [PYR, PV, SOM, VIP], sorted by PYR activity.
    """

    # --- Step 1: Generate all combinations of initial values for the four populations ---
    all_init_values = [
        0.001,
        0.1,
        0.3,
        0.5,
    ]  # Very Low, Low, Medium, High firing rates
    all_init_values_per_pop = [all_init_values] * 4  # same grid for all populations

    # Cartesian product of initial values for all populations
    # Test all combinations of initial conditions
    x0 = np.array([np.array(items) for items in product(*all_init_values_per_pop)])

    # --- Step 2: Prepare the parameter list for the equations ---
    parameter_list = (
        w_11,
        w_21,
        w_31,
        w_12,
        w_32,
        w_42,
        w_22,
        w_13,
        w_23,
        w_43,
        w_33,
        w_14,
        w_24,
        w_34,
        w_44,
        i_pyr,
        i_inter_pv,
        i_inter_som,
        i_inter_vip,
        J_r1,
    )

    # --- Step 3: Solve for steady states using fsolve ---
    paramSolve = []
    for i in range(np.size(x0, 0)):
        sol, infodict, ier, mesg = fsolve(
            equations,
            x0[i, :],
            args=parameter_list,
            fprime=equations_prime,
            full_output=True,
            xtol=1e-12,
        )
        if ier == 1:  # successful convergence
            paramSolve.append(sol)

    if len(paramSolve) == 0:
        return np.array([])  # no solutions found

    paramSolve = np.array(paramSolve)

    # --- Step 4: Remove duplicate steady states ---
    # Round solutions to avoid floating-point duplicates
    paramSolve_rounded = np.round(paramSolve * 1e6)
    b = np.ascontiguousarray(paramSolve_rounded).view(
        np.dtype(
            (np.void, paramSolve_rounded.dtype.itemsize * paramSolve_rounded.shape[1])
        )
    )
    _, unique_indices = np.unique(b, return_index=True)

    # --- Step 5: Classify each unique steady state ---
    stable_indices = []
    unstable_indices = []
    stable_indices_w = []
    unstable_indices_w = []
    all_w = []
    all_wi = []

    for idx in unique_indices:
        # Compute Jacobian at the steady state
        M = equations_prime(paramSolve[idx], *parameter_list)
        w, v = LA.eig(M)
        wi = w.imag
        w = w.real
        all_w.append(w)
        all_wi.append(wi)

        # Check stability: all real parts of eigenvalues <= 0
        if np.all(w <= 0):
            stable_indices.append(idx)
            stable_indices_w.append(len(stable_indices_w))
        else:
            unstable_indices.append(idx)
            unstable_indices_w.append(len(unstable_indices_w))

    # --- Step 6: Sort and organize results ---
    stable_paramSolve = paramSolve[stable_indices]
    stable_sort_idx = stable_paramSolve[:, 0].argsort()
    stable_paramSolve = stable_paramSolve[stable_sort_idx]

    unstable_paramSolve = paramSolve[unstable_indices]
    unstable_sort_idx = unstable_paramSolve[:, 0].argsort()
    unstable_paramSolve = unstable_paramSolve[unstable_sort_idx]

    return stable_paramSolve


def run_simulation(max_states: int, stable_paramSolve: np.ndarray):
    """
    Simulate the dynamics of four interacting neural populations until
    a specified number of state transitions (UP <-> DOWN) occurs.

    Args:
        max_states (int): Maximum number of UP/DOWN transitions to simulate.
        stable_paramSolve (np.ndarray): Array of stable steady states
                                        [PYR, PV, SOM, VIP].

    Returns:
        R (np.ndarray): Array of firing rates over time (shape: time x 4).
        all_periods_up_states (list): List of durations of UP states.
        all_periods_down_states (list): List of durations of DOWN states.
        up_state_timestamps (list): List of time indices marking UP state onset.
        down_state_timestamps (list): List of time indices marking DOWN state onset.
    """

    # --- Initialize firing rate arrays and adaptation currents ---
    R = np.zeros((int(tmax / dt), 4))
    i_adapt_pyr = np.zeros(int(tmax / dt))
    i_adapt_som = np.zeros(int(tmax / dt))

    # Set initial conditions from the first stable steady state
    R[0, 0] = stable_paramSolve[0][0] * fact_pyr
    R[0, 1] = stable_paramSolve[0][1] * fact_pv
    R[0, 2] = stable_paramSolve[0][2] * fact_som
    R[0, 3] = stable_paramSolve[0][3] * fact_vip
    i_adapt_pyr[0] = J_r1 * stable_paramSolve[0][0]

    # --- Time vector and noise term ---
    time = np.arange(0, tmax, dt)
    D = sigma**2 / (2 * tau_s)  # noise intensity

    # --- Variables for state tracking ---
    state_nb = 0  # 0 = DOWN, 1 = UP
    index_tau = 0
    i = 0

    all_periods_up_states = []
    all_periods_down_states = []
    up_state_timestamps = []
    down_state_timestamps = []

    # --- Simulation loop ---
    while len(all_periods_up_states) < max_states:

        # --- Compute inputs for each population ---
        # PYR inputs
        x_pyr = w_11 * R[i, 0] / fact_pyr + i_pyr
        y_pyr = (
            w_31 * R[i, 2] / fact_som
            + (1 - kd) * w_21 * R[i, 1] / fact_pv
            + i_adapt_pyr[i]
        )
        z_pyr = kd * w_21 * R[i, 1] / fact_pv

        # Update PYR firing rate
        R[i + 1, 0] = Phi_heav(
            -R[i, 0] / tau_s * dt
            + (fact_pyr * k_input_output(z_pyr, alpha_pyr, theta_pyr) - R[i, 0])
            * Phi_input_output(x_pyr, y_pyr, z_pyr, alpha_pyr, theta_pyr)
            / tau_s
            * dt
            + R[i, 0]
            + fact_pyr * np.sqrt(2 * D) * np.sqrt(dt) * np.random.randn()
        )

        # Update PYR adaptation current
        i_adapt_pyr[i + 1] = (
            -i_adapt_pyr[i] / tau_a_1 * dt
            + R[i, 0] / fact_pyr * J_r1 / tau_a_1 * dt
            + i_adapt_pyr[i]
        )

        # PV inputs and update
        x_pv = w_12 * R[i, 0] / fact_pyr + i_inter_pv
        y_pv = (
            w_32 * R[i, 2] / fact_som
            + w_22 * R[i, 1] / fact_pv
            + w_42 * R[i, 3] / fact_vip
        )
        z_pv = 0
        R[i + 1, 1] = Phi_heav(
            -R[i, 1] / tau_s * dt
            + (fact_pv * k_input_output(z_pv, alpha_inter_pv, theta_inter_pv) - R[i, 1])
            * Phi_input_output(x_pv, y_pv, z_pv, alpha_inter_pv, theta_inter_pv)
            / tau_s
            * dt
            + R[i, 1]
            + fact_pyr * np.sqrt(2 * D) * np.sqrt(dt) * np.random.randn()
        )

        # SOM inputs and update
        x_som = w_13 * R[i, 0] / fact_pyr + i_inter_som
        y_som = (
            w_23 * R[i, 1] / fact_pv
            + w_33 * R[i, 2] / fact_som
            + w_43 * R[i, 3] / fact_vip
        )
        z_som = 0
        R[i + 1, 2] = Phi_heav(
            -R[i, 2] / tau_s * dt
            + (
                fact_som * k_input_output(z_som, alpha_inter_som, theta_inter_som)
                - R[i, 2]
            )
            * Phi_input_output(x_som, y_som, z_som, alpha_inter_som, theta_inter_som)
            / tau_s
            * dt
            + R[i, 2]
            + fact_pyr * np.sqrt(2 * D) * np.sqrt(dt) * np.random.randn()
        )

        # VIP inputs and update
        x_vip = w_14 * R[i, 0] / fact_pyr + i_inter_vip
        y_vip = (
            w_24 * R[i, 1] / fact_pv
            + w_34 * R[i, 2] / fact_som
            + w_44 * R[i, 3] / fact_vip
        )
        z_vip = 0
        R[i + 1, 3] = Phi_heav(
            -R[i, 3] / tau_s * dt
            + (
                fact_vip * k_input_output(z_vip, alpha_inter_vip, theta_inter_vip)
                - R[i, 3]
            )
            * Phi_input_output(x_vip, y_vip, z_vip, alpha_inter_vip, theta_inter_vip)
            / tau_s
            * dt
            + R[i, 3]
            + fact_pyr * np.sqrt(2 * D) * np.sqrt(dt) * np.random.randn()
        )

        # --- State transitions tracking ---
        if state_nb == 0:  # currently DOWN
            if R[i + 1, 0] > stable_paramSolve[1][0] * fact_pyr:
                state_nb = 1
                if index_tau != 0:
                    period = i - index_tau
                    if (
                        len(all_periods_down_states) > 0
                        and time[period] > 200
                        and all_periods_down_states[-1] > 200
                    ):
                        print("Error: down state stability is too high")
                        break
                    all_periods_down_states.append(time[period])
                    down_state_timestamps.append(i)
                index_tau = i
        elif state_nb == 1:  # currently UP
            if R[i + 1, 0] < stable_paramSolve[0][0] * fact_pyr:
                state_nb = 0
                if index_tau != 0 and len(all_periods_down_states) > 0:
                    period = i - index_tau
                    if (
                        len(all_periods_up_states) > 0
                        and time[period] > 200
                        and all_periods_up_states[-1] > 200
                    ):
                        print("Error: up state stability is too high")
                        break
                    all_periods_up_states.append(time[period])
                    up_state_timestamps.append(i)
                index_tau = i

        # Safety check for very long states
        if time[i - index_tau] > 500:
            print(f"Error: state {state_nb} (0=down, 1=up) duration exceeds 500 sec")
            break

        i += 1

    return (
        R,
        all_periods_up_states,
        all_periods_down_states,
        up_state_timestamps,
        down_state_timestamps,
    )


stable_paramSolve = find_steady_states()
print("Stable steady states found (PYR, PV, SOM, VIP):", stable_paramSolve)
res = run_simulation(max_states=10, stable_paramSolve=stable_paramSolve)
(
    R,
    all_periods_up_states,
    all_periods_down_states,
    up_state_timestamps,
    down_state_timestamps,
) = res

print("Mean UP state duration", np.mean(all_periods_up_states))
print("Mean DOWN state duration", np.mean(all_periods_down_states))
print(
    "Mode UP state duration",
    scipy.stats.mode(all_periods_up_states, keepdims=True).mode.item(),
)
print(
    "Mode DOWN state duration",
    scipy.stats.mode(all_periods_down_states, keepdims=True).mode.item(),
)

assert down_state_timestamps[0] < up_state_timestamps[0]


def up_timestamp(i):
    if i == -1:
        return 0
    else:
        return up_state_timestamps[i]


up_pyr = np.concatenate(
    [
        R[down_state_timestamps[idx] : up_state_timestamps[idx], 0]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

down_pyr = np.concatenate(
    [
        R[up_timestamp(idx - 1) : down_state_timestamps[idx], 0]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

up_pv = np.concatenate(
    [
        R[down_state_timestamps[idx] : up_state_timestamps[idx], 1]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

down_pv = np.concatenate(
    [
        R[up_timestamp(idx - 1) : down_state_timestamps[idx], 1]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

up_som = np.concatenate(
    [
        R[down_state_timestamps[idx] : up_state_timestamps[idx], 2]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

down_som = np.concatenate(
    [
        R[up_timestamp(idx - 1) : down_state_timestamps[idx], 2]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

up_vip = np.concatenate(
    [
        R[down_state_timestamps[idx] : up_state_timestamps[idx], 3]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)

down_vip = np.concatenate(
    [
        R[up_timestamp(idx - 1) : down_state_timestamps[idx], 3]
        for idx in range(len(up_state_timestamps) - 1)
    ]
)


print("Median PYR firing rate in UP states", np.median(up_pyr))
print("Median PYR firing rate in DOWN states", np.median(down_pyr))
print("Median PV firing rate in UP states", np.median(up_pv))
print("Median PV firing rate in DOWN states", np.median(down_pv))
print("Median SOM firing rate in UP states", np.median(up_som))
print("Median SOM firing rate in DOWN states", np.median(down_som))
print("Median VIP firing rate in UP states", np.median(up_vip))
print("Median VIP firing rate in DOWN states", np.median(down_vip))

# Plot the firing rate of PYR population
# drop rows that are all zeros or all-NaN
mask = ~(np.all(R == 0, axis=1) | np.all(np.isnan(R), axis=1))
if not np.any(mask):
    print("Warning: all R rows are empty or zero; nothing to plot.")
else:
    R = R[mask]
plt.figure(figsize=(10, 5))
plt.plot(
    range(len(R[:, 0])), R[:, 0], color="blue", label="PYR Firing Rate", markersize=0.2
)
plt.title("Firing Rate of PYR Population Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Firing Rate")
plt.legend()
plt.grid()
plt.show()

# plot the smoothed firing rate of PYR population
from scipy.ndimage import gaussian_filter1d

smoothed_pyr = gaussian_filter1d(R[:, 0], sigma=100)
plt.figure(figsize=(10, 5))
plt.plot(
    range(len(smoothed_pyr)),
    smoothed_pyr,
    color="red",
    label="Smoothed PYR Firing Rate",
    markersize=0.2,
)
plt.title("Smoothed Firing Rate of PYR Population Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Firing Rate")
plt.legend()
plt.grid()
plt.show()
