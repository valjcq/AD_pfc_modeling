"""
FIRING RATE NETWORK MODEL FOR DECAYING BUMP

No recurrent excitation.
E-cells include a slow activity-dependent depolarizing current (Im).
"""

import numpy as np
import matplotlib.pyplot as plt

# ===== PARAMETERS =====

N    = 512  # number of "neurons" in each population of the rate model
npop = 8    # number of cues presented to the network

totalTime = 4200  # total time of the simulation in ms
dt = 2            # integration step in ms

tauE  = 20   # time constant of rate equation for excitatory neurons
tauI  = 10   # time constant of rate equation for inhibitory neurons
tauIm = 300  # time constant of activity-dependent depolarizing current Im
aIm   = 0.85 # rate of activation of activity-dependent depolarizing current Im

GEE = 0    # strength of excitation to excitatory neurons (no recurrent E-to-E)
GEI = 4    # strength of excitation to inhibitory neurons
GIE = 2    # strength of inhibition to excitatory neurons
GII = 1    # strength of inhibition to inhibitory neurons

I0E = 0.6   # external bias current to excitatory neurons
I0I = 0.28  # external bias current to inhibitory neurons

sigE = 5  # standard deviation of additive noise in rate equation of e-cells
sigI = 3  # standard deviation of additive noise in rate equation of i-cells

kappa    = 1     # parameter defining concentration of input to e-cells
stimon   = 1000  # time when external stimulus is applied in ms
stimoff  = 1500  # time when external stimulus ceases in ms
stim     = 4500  # strength of external stimulus
delayend = 3500  # time when delay ends in ms

# ===== PRELIMINARY CALCULATIONS =====

rE = np.zeros(N)
rI = np.zeros(N)
Im = np.zeros(N)
nsteps   = int(totalTime / dt)
delayPop = np.zeros(N)

# No recurrent E-to-E connectivity, only autapses allowed (GEE=0 so unused)
WE = np.eye(N)

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

    # Current input to each population (GEE=0, WE term vanishes)
    IE = GEE * (WE @ rE) + (I0E - GIE * np.mean(rI)) * np.ones(N)
    II = (GEI * np.mean(rE) - GII * np.mean(rI) + I0I) * np.ones(N)

    # External task-dependent inputs
    if stimon_step < i < stimoff_step:
        IE = IE + stimulus  # cue stimulus before delay
    if delayend_step < i < delayend_step + (stimoff_step - stimon_step):
        IE = IE - 2000 * stim  # erasing global input after delay

    if delayend_step - delaywin < i <= delayend_step:
        delayPop += rE / delaywin

    # Euler integration
    rE = rE + (f(IE) + Im - rE + noiseE) * dt / tauE
    rI = rI + (f(II) - rI + noiseI) * dt / tauI
    Im = Im + (aIm * rE / (1 + np.exp(-2 * (rE - 2))) - Im) * dt / tauIm

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
ax1.axvline(stimon,   color='cyan',  lw=1.5, ls='--', label='stim on/off')
ax1.axvline(stimoff,  color='cyan',  lw=1.5, ls='--')
ax1.axvline(delayend, color='white', lw=1.5, ls='--', label='delay end')
ax1.plot(history_t, np.degrees(history_ang), 'g.', ms=1.5, label='decoded angle')
ax1.set_ylabel('neuron (deg)')
ax1.set_title('Decaying Bump — e-cell activity')
ax1.legend(loc='upper right', fontsize=7)

# Middle panel: e-cell profile at end of delay
ax2 = axes[1]
delay_idx = np.searchsorted(history_t, delayend) - 1
ax2.plot(theta_deg, history_rE[delay_idx], 'r', label=f't={delayend} ms (end of delay)')
ax2.plot(np.degrees(history_ang[delay_idx]), 14, 'kv', ms=8, markerfacecolor='k')
ax2.set_xlim([theta_deg[0], theta_deg[-1]])
ax2.set_ylim([0, 15])
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

plt.suptitle('Decaying Bump')
plt.tight_layout()
plt.savefig('decaying_bump_final_state.png', dpi=300)
plt.show()

print(f"Decoded response angle at end of delay: {np.degrees(response):.1f} deg")
