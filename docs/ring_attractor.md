# Ring Attractor Network - Model and Implementation

This document describes the model equations and implementation of the ring attractor.

For experiment protocols and outputs, see `docs/ring_experiments.md`.

## 1. Architecture

- `N` nodes arranged on a ring (default often `128`).
- Each node contains a 4-population local circuit: PYR, PV, SOM, VIP.
- Inter-node coupling:
- PYR -> PYR: distance-dependent Gaussian profile.
- PV -> PYR: global inhibition.
- SOM/VIP: local only.

## 2. Connectivity

### 2.1 Angular distance

`d(theta_i, theta_j) = min(|theta_i-theta_j|, 2*pi-|theta_i-theta_j|)`.

### 2.2 PYR -> PYR

- Raw Gaussian profile over ring distance.
- Row-sum normalization ensures total PYR coupling equals `w_pyr_pyr_inter` for each node.

### 2.3 PV -> PYR

- Uniform all-to-all inhibition (excluding self) with total strength `w_pv_global`.

## 3. Local Dynamics

Each population rate follows a first-order rate equation with transfer function and optional noise.

- Inputs combine local recurrent terms, inter-node terms (for PYR), adaptation (PYR/SOM), and external currents.
- Rates are bounded to a finite safe range in simulation.

## 4. Adaptation

PYR and SOM include adaptation currents with independent strengths and time constants.

## 5. Transfer Function

- Wong-Wang style nonlinear transfer function.
- Population-specific threshold and gain parameters.

## 6. Stimulus Protocol

Active ring experiments use a cue + delay structure:

- Burn-in (network settles)
- Pre-cue baseline
- Cue stimulus (Gaussian over angle)
- Delay period (memory maintenance)
- Optional response transient

## 7. Noise Modes

- `none`
- `white`
- `ou`

`ring-diffusion` uses delay trajectories and MSD-based analysis under these simulation modes.

## 8. Conditions

Eight receptor conditions are supported in the ring model:

- `WT`, `WT_APP`
- `a7_KO`, `a7_KO_APP`
- `b2_KO`, `b2_KO_APP`
- `a5_KO`, `a5_KO_APP`

## 9. Active Ring Experiment Family

Current maintained ring experiments are:

- `ring-run`
- `ring-study`
- `ring-diffusion`
- `ring-noise-floor`
- `ring-calibrate`
- `ring-asymmetry`
- `ring-burnin-stability`
