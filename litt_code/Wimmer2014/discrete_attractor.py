"""
FIRING RATE NETWORK MODEL FOR DISCRETE ATTRACTOR

Discrete piece-wise connectivity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant

# ===== PARAMETERS =====

N    = 512  # number of "neurons" in each population of the rate model
npop = 8    # number of populations in the network

totalTime = 4200  # total time of the simulation in ms
dt = 2            # integration step in ms

tauE = 20   # time constant of rate equation for excitatory neurons
tauI = 10   # time constant of rate equation for inhibitory neurons
GEE = 2.9   # strength of excitation to excitatory neurons
GEI = 4     # strength of excitation to inhibitory neurons
GIE = 1.2   # strength of inhibition to excitatory neurons
GII = 1.3   # strength of inhibition to inhibitory neurons

I0E = -1.2  # external bias current to excitatory neurons
I0I = 0.28  # external bias current to inhibitory neurons

sigE = 12  # standard deviation of additive noise in rate equation of e-cells
sigI = 3   # standard deviation of additive noise in rate equation of i-cells

kappa    = 3     # parameter defining concentration of input to e-cells
stimon   = 1000  # time when external stimulus is applied in ms
stimoff  = 1500  # time when external stimulus ceases in ms
stim     = 250   # strength of external stimulus
delayend = 3500  # time when delay ends in ms

# ===== PRELIMINARY CALCULATIONS =====

rE = np.zeros(N)
rI = np.zeros(N)
nsteps   = int(totalTime / dt)
delayPop = np.zeros(N)

# E-to-E connectivity: block-structured circulant (discrete populations)
nbl = N // npop

th8  = np.arange(npop) / npop * 2 * np.pi
kap8 = 4
v    = np.exp(kap8 * np.cos(th8))
v    = v / np.sum(v)

# Build npop x npop circulant, expand to N x N via Kronecker product
# circulant(v) is equivalent to MATLAB gallery('circul',v) since v is symmetric
WEsm = circulant(v)                               # npop x npop
WE   = np.kron(WEsm, np.ones((nbl, nbl))) / nbl  # N x N

# Equivalent to MATLAB circshift(WE, [nbl/2, nbl/2])
WE = np.roll(np.roll(WE, nbl // 2, axis=0), nbl // 2, axis=1)

# All-to-all inhibitory connections (uniform mean-field)
# WEI = WIE = WII = ones(N)/N -> equivalent to using np.mean()

# Stimulus parameters
theta = (np.arange(1, N + 1) - 0.5) / N * 2 * np.pi
theta = theta - np.pi

stimulus = np.exp(kappa * np.cos(theta))
stimulus = stim * stimulus / np.sum(stimulus)

stimon_step   = int(stimon / dt)
stimoff_step  = int(stimoff / dt)
delayend_step = int(delayend / dt)
delaywin      = int(100 / dt)  # 100 ms window

# Input-output function (Brunel, Cereb Cortex 13:1151, 2003)
def f(x):
    return (x**2 * (x > 0) * (x < 1) +
            np.sqrt(np.maximum(4 * x - 3, 0)) * (x >= 1))

# Population vector decoder
def decode(r, th):
    return np.arctan2(np.sum(r * np.sin(th)), np.sum(r * np.cos(th)))

# Indices for piecewise coloring (odd/even populations)
vect    = np.arange(1, N + 1)
indodd  = np.where(np.mod(np.ceil((vect + nbl / 2) / nbl), 2) == 1)[0]
indeven = np.where(np.mod(np.ceil((vect + nbl / 2) / nbl), 2) == 0)[0]

theta2 = theta.copy()
# np.diff produces len-1 array; use np.where to get integer indices into indodd/indeven
theta2[indodd[np.where(np.diff(indodd) > 1)[0]]]   = np.nan
theta2[indeven[np.where(np.diff(indeven) > 1)[0]]] = np.nan

# ===== SIMULATION LOOP =====

# Record history every 10 steps (20 ms)
record_every = 10
n_record = nsteps // record_every
history_rE  = np.zeros((n_record, N))
history_rI  = np.zeros((n_record, N))
history_ang = np.zeros(n_record)
history_t   = np.zeros(n_record)

response = 0.0
ang = 0.0
rec_idx = 0

for i in range(1, nsteps + 1):

    # Additive noise for each population
    noiseE = sigE * np.random.randn(N)
    noiseI = sigI * np.random.randn(N)

    # Current input to each population
    # WEI = WIE = WII = ones(N)/N, so W*r = mean(r)*ones(N)
    IE = GEE * (WE @ rE) + I0E * np.ones(N) - GIE * np.mean(rI) * np.ones(N)
    II = GEI * np.mean(rE) * np.ones(N) - GII * np.mean(rI) * np.ones(N) + I0I * np.ones(N)

    # External task-dependent inputs
    if stimon_step < i < stimoff_step:
        IE = IE + stimulus  # cue stimulus before delay
    if delayend_step < i < delayend_step + (stimoff_step - stimon_step):
        IE = IE - stim      # erasing global input after delay

    if delayend_step - delaywin < i <= delayend_step:
        delayPop += rE / delaywin

    # Euler integration
    rE = rE + (f(IE) - rE + noiseE) * dt / tauE
    rI = rI + (f(II) - rI + noiseI) * dt / tauI

    # Decoded angle from network activity
    ang = decode(rE, theta)
    if i < delayend_step:
        response = ang

    # Record
    if i % record_every == 0:
        history_rE[rec_idx]  = rE
        history_rI[rec_idx]  = rI
        history_ang[rec_idx] = ang
        history_t[rec_idx]   = i * dt
        rec_idx += 1

# ===== PLOT =====

theta_deg = np.degrees(theta)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.patch.set_facecolor('white')

# Top panel: e-cell activity heatmap (neuron x time)
ax1 = axes[0]
im = ax1.imshow(
    history_rE.T,
    aspect='auto',
    origin='lower',
    extent=[history_t[0], history_t[-1], theta_deg[0], theta_deg[-1]],
    cmap='hot',
    vmin=0
)
plt.colorbar(im, ax=ax1, label='e-cell rate')
ax1.axvline(stimon,   color='cyan',  lw=1.5, ls='--', label='stim on/off')
ax1.axvline(stimoff,  color='cyan',  lw=1.5, ls='--')
ax1.axvline(delayend, color='white', lw=1.5, ls='--', label='delay end')
ax1.plot(history_t, np.degrees(history_ang), 'g.', ms=1.5, label='decoded angle')
ax1.set_ylabel('neuron (deg)')
ax1.set_title('Discrete Attractor — e-cell activity')
ax1.legend(loc='upper right', fontsize=7)

# Middle panel: e-cell profile at end of delay (piecewise coloring)
ax2 = axes[1]
delay_idx = np.searchsorted(history_t, delayend) - 1
rE_delay = history_rE[delay_idx]
ax2.plot(theta2[indeven], rE_delay[indeven], 'r')
ax2.plot(theta2[indodd],  rE_delay[indodd],  color=(0.75, 0, 0))
ax2.plot(np.degrees(history_ang[delay_idx]), 11 / 12 * 18, 'kv', ms=8, markerfacecolor='k')
ax2.set_xlim([theta_deg[0], theta_deg[-1]])
ax2.set_ylim([0, 18])
ax2.set_ylabel('e-cell rate')
ax2.set_xlabel('neuron (deg)')
ax2.set_title(f't={delayend} ms (end of delay)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Bottom panel: i-cell profile at end of delay
ax3 = axes[2]
ax3.plot(theta_deg, history_rI[delay_idx], 'b')
ax3.set_xlim([theta_deg[0], theta_deg[-1]])
ax3.set_ylim([0, 10])
ax3.set_ylabel('i-cell rate')
ax3.set_xlabel('neuron (deg)')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.suptitle('Discrete Attractor')
plt.tight_layout()
plt.savefig('discrete_attractor_final_state.png', dpi=300)
plt.show()

print(f"Decoded response angle at end of delay: {np.degrees(response):.1f} deg")
