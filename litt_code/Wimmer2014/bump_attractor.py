"""
FIRING RATE NETWORK MODEL FOR BUMP ATTRACTOR

Continuous ring connectivity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant

# ===== PARAMETERS =====

N = 512       # number of "neurons" in the rate model
npop = 8      # number of cues presented

totalTime = 4200  # total time of the simulation in ms
dt = 2            # integration step in ms

tauE = 20  # time constant of rate equation for excitatory neurons
tauI = 10  # time constant of rate equation for inhibitory neurons

kappa = 1.5  # parameter defining concentration of e-to-e connectivity
GEE = 6      # strength of excitation to excitatory neurons
GEI = 4      # strength of excitation to inhibitory neurons
GIE = 3.4    # strength of inhibition to excitatory neurons
GII = 0.85   # strength of inhibition to inhibitory neurons

I0E = 0.2  # external bias current to excitatory neurons
I0I = 0.5  # external bias current to inhibitory neurons

sigE = 1  # standard deviation of additive noise in rate equation of e-cells
sigI = 3  # standard deviation of additive noise in rate equation of i-cells

stimon   = 1000  # time when external stimulus is applied in ms
stimoff  = 1500  # time when external stimulus ceases in ms
stim     = 200   # strength of external stimulus
delayend = 3500  # time when delay ends in ms

# Distractor parameters (applied during delay period)
distractoron  = 2000   # time when distractor is applied in ms
distractoroff = 2500   # time when distractor ceases in ms
distractor_stim   = 0   # strength of distractor
distractor_angle  = 150    # angular offset of distractor from cue (degrees)

# ===== PRELIMINARY CALCULATIONS =====

rE = np.zeros(N)
rI = np.zeros(N)
nsteps   = int(totalTime / dt)
delayPop = np.zeros(N)

# E-to-E connectivity: circulant matrix with von Mises kernel
theta = np.arange(N) / N * 2 * np.pi
v = np.exp(kappa * np.cos(theta))
v = v / np.sum(v)
WE = circulant(v)  # equivalent to MATLAB gallery('circul', v) since v is symmetric

# Stimulus parameters
theta = theta - np.pi
v = np.exp(kappa * np.cos(theta))
v = v / np.sum(v)
stimulus = stim * v

stimon_step   = int(stimon / dt)
stimoff_step  = int(stimoff / dt)
delayend_step = int(delayend / dt)
delaywin      = int(100 / dt)  # 100 ms window

# Distractor stimulus kernel: same shape as cue but at a different angle
distractor_offset = np.radians(distractor_angle)
v_dist = np.exp(kappa * np.cos(theta - distractor_offset))
v_dist = v_dist / np.sum(v_dist)
distractor = distractor_stim * v_dist

distractoron_step  = int(distractoron / dt)
distractoroff_step = int(distractoroff / dt)

# Input-output function (Brunel, Cereb Cortex 13:1151, 2003)
def f(x):
    return (x**2 * (x > 0) * (x < 1) +
            np.sqrt(np.maximum(4 * x - 3, 0)) * (x >= 1))

# Population vector decoder
def decode(r, th):
    return np.arctan2(np.sum(r * np.sin(th)), np.sum(r * np.cos(th)))

# ===== SIMULATION LOOP =====

# Record history every 10 steps (20 ms)
record_every = 10
n_record = nsteps // record_every
history_rE = np.zeros((n_record, N))
history_rI = np.zeros((n_record, N))
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
    IE = GEE * (WE @ rE) + (I0E - GIE * np.mean(rI)) * np.ones(N)
    II = (GEI * np.mean(rE) - GII * np.mean(rI) + I0I) * np.ones(N)

    # External task-dependent inputs
    if stimon_step < i < stimoff_step:
        IE = IE + stimulus  # cue stimulus before delay
    if distractoron_step < i < distractoroff_step:
        IE = IE + distractor  # distractor during delay
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
        history_rE[rec_idx] = rE
        history_rI[rec_idx] = rI
        history_ang[rec_idx] = ang
        history_t[rec_idx] = i * dt
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
ax1.axvline(stimon,        color='cyan',   lw=1.5, ls='--', label='stim on/off')
ax1.axvline(stimoff,       color='cyan',   lw=1.5, ls='--')
ax1.axvline(distractoron,  color='orange', lw=1.5, ls='--', label='distractor on/off')
ax1.axvline(distractoroff, color='orange', lw=1.5, ls='--')
ax1.axvline(delayend,      color='white',  lw=1.5, ls='--', label='delay end')
ax1.plot(history_t, np.degrees(history_ang), 'g.', ms=1.5, label='decoded angle')
ax1.set_ylabel('neuron (deg)')
ax1.set_title('Bump Attractor — e-cell activity')
ax1.legend(loc='upper right', fontsize=7)

# Middle panel: e-cell profile at end of delay
ax2 = axes[1]
delay_idx = np.searchsorted(history_t, delayend) - 1
ax2.plot(theta_deg, history_rE[delay_idx], 'r', label=f't={delayend} ms (end of delay)')
ax2.plot(np.degrees(history_ang[delay_idx]), 11, 'kv', ms=8, markerfacecolor='k')
ax2.set_xlim([theta_deg[0], theta_deg[-1]])
ax2.set_ylim([0, 12])
ax2.set_ylabel('e-cell rate')
ax2.set_xlabel('neuron (deg)')
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Bottom panel: i-cell profile at end of delay
ax3 = axes[2]
ax3.plot(theta_deg, history_rI[delay_idx], 'b', label=f't={delayend} ms (end of delay)')
ax3.set_xlim([theta_deg[0], theta_deg[-1]])
ax3.set_ylim([0, 10])
ax3.set_ylabel('i-cell rate')
ax3.set_xlabel('neuron (deg)')
ax3.legend()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.suptitle(f'Bump Attractor — distractor at {distractor_angle}° offset')
plt.tight_layout()
plt.savefig('bump_attractor_final_state.png', dpi=300)
plt.show()

print(f"Decoded response angle at end of delay: {np.degrees(response):.1f} deg")
