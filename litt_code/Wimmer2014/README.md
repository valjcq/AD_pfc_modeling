# Wimmer et al. (2014) — Three Working Memory Network Models

These three firing-rate models all simulate the same delayed-response task
(stimulus → delay → readout) but use fundamentally different mechanisms for
maintaining the memory trace across the delay period.

---

## Task structure (common to all three)

| Phase | Time (ms) | What happens |
|---|---|---|
| Baseline | 0 – 1000 | Network at rest |
| Stimulus | 1000 – 1500 | Localized cue drives E-cells |
| Delay | 1500 – 3500 | Network must maintain the memory without input |
| Readout | 3500 – 4200 | Global erasing input applied; response decoded |

All models share the same E/I architecture (512 E-cells + 512 I-cells arranged
on a ring), the same transfer function, and the same Euler integration scheme.

---

## 1. Bump Attractor (`bump_attractor.py`)

**Core mechanism:** Recurrent E-to-E excitation with a smooth von Mises
connectivity kernel keeps the network in a self-sustaining "bump" of activity.

**Connectivity:**
```
WE = circulant(exp(κ · cos(θ)))   κ = 1.5   GEE = 6
```
The weight matrix is a *continuous* circulant — every neuron is connected to
every other with a weight that decays smoothly with angular distance.

**E-cell update:**
```
τE · ṙE = f(GEE·WE·rE + I0E − GIE·<rI>) − rE + noise
```

**Memory mechanism:** The bump is a true *attractor state*. Recurrent
excitation is strong enough that, once seeded by the stimulus, the network
self-sustains the bump throughout the delay with no external input.

**Key property:** Continuous family of attractors — the bump can sit at *any*
angle, so the memory is analogue and graded. Small noise causes the decoded
angle to drift slowly ("diffusion").

**Parameters snapshot:**

| Param | Value |
|---|---|
| GEE | 6 (strong recurrent E↔E) |
| GEI / GIE / GII | 4 / 3.4 / 0.85 |
| κ (connectivity width) | 1.5 |
| sigE / sigI (noise) | 1 / 3 |

---

## 2. Discrete Attractor (`discrete_attractor.py`)

**Core mechanism:** The ring is divided into `npop = 8` discrete populations.
E-to-E connectivity is block-structured so only neurons within or near the
same population strongly excite each other.

**Connectivity:**
```
WEsm = circulant(exp(κ8 · cos(θ8)))   κ8 = 4   (8-point ring)
WE   = kron(WEsm, ones(nbl, nbl)) / nbl          GEE = 2.9
```
The weight matrix is a *staircase* (block Kronecker product) — sharper than
the continuous case, discretising the ring into 8 equally-spaced bumps.

**E-cell update:** Same form as bump attractor but with the block WE.

**Memory mechanism:** The network has only **8 discrete stable states** (one
per population). The stimulus pushes the network toward the nearest discrete
state, where it locks in. Memory is robust to noise because the bump cannot
drift continuously — it would have to jump discretely.

**Key property:** Categorical memory. The decoded angle snaps to one of 8
positions. Noise (`sigE = 12`, much higher than other models) does not cause
drift but can cause rare discrete jumps between states.

**Parameters snapshot:**

| Param | Value |
|---|---|
| GEE | 2.9 |
| GEI / GIE / GII | 4 / 1.2 / 1.3 |
| κ8 (connectivity width, 8-pop ring) | 4 (narrow = discrete) |
| sigE / sigI (noise) | 12 / 3 (high noise tolerated) |
| I0E | −1.2 (negative bias → needs bump to activate) |

---

## 3. Decaying Bump (`decaying_bump.py`)

**Core mechanism:** **No recurrent E-to-E excitation** (`GEE = 0`). Memory is
maintained by a slow intrinsic current `Im` — an activity-dependent
depolarising current (analogous to a persistent sodium or calcium current) that
activates during the stimulus and then slowly decays.

**Connectivity:**
```
WE = eye(N)   GEE = 0   (autapses only, effectively unused)
```

**E-cell update (with intrinsic current):**
```
τE  · ṙE = f(I0E − GIE·<rI> + Im) − rE + noise
τIm · İm = aIm · rE / (1 + exp(−2(rE − 2))) − Im
```
`Im` has a long time constant (`τIm = 300 ms`) and activates only when `rE`
is high (sigmoidal gate), so it is recruited during the stimulus and persists
into the delay.

**Memory mechanism:** The bump *decays* over the delay period as `Im` slowly
winds down — hence the name. There is no stable attractor; the bump amplitude
monotonically decreases. The angular position is preserved while activity
lasts, but precision degrades over time.

**Key property:** No persistent attractor. Memory fidelity decreases with
delay duration. This models a regime where working memory relies on
*intrinsic cellular* rather than *network* mechanisms.

**Parameters snapshot:**

| Param | Value |
|---|---|
| GEE | 0 (no recurrent excitation) |
| GEI / GIE / GII | 4 / 2 / 1 |
| τIm | 300 ms (slow intrinsic current) |
| aIm | 0.85 (activation strength) |
| sigE / sigI (noise) | 5 / 3 |

---

## Summary comparison

| Feature | Bump Attractor | Discrete Attractor | Decaying Bump |
|---|---|---|---|
| E-to-E recurrence | Continuous (von Mises) | Block / discrete | None (GEE = 0) |
| Memory mechanism | Network attractor | Network attractor | Intrinsic current Im |
| Stable states | Continuous (any angle) | 8 discrete positions | None (decays) |
| Noise robustness | Low (diffusion) | High (discrete jumps) | Low (fades) |
| Drift during delay | Yes (analogue diffusion) | No (categorical lock) | Angle stable, amplitude decays |
| GEE | 6 | 2.9 | 0 |
| Distractor handling | Supported | Not included | Not included |
| Biological analogue | Cortical ring / PFC | Categorical WM | Persistent currents (NaP, Ca2+) |

---

## Reference

Wimmer, K., Nykamp, D. Q., Compte, A., & Roxin, A. (2014).
*Bump attractor dynamics in prefrontal cortex explains behavioral precision in
spatial working memory.* Nature Neuroscience, 17(3), 431–439.
https://doi.org/10.1038/nn.3645
