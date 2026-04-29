# Single-Node to Ring Weight Conversion

**Status:** Design proposal — not yet implemented or tested.

The core idea: the single-node optimization produces fitted weights that represent the *total* synaptic drive onto each population. When moving to the ring network, this total drive should be preserved at the homogeneous fixed point by distributing it spatially according to a connectivity kernel. Fixed-point equivalence is guaranteed by construction via row-sum normalization.

---

## General Principle

At a homogeneous fixed point (uniform ring, no bump), all nodes fire identically. Any spatially distributed weight matrix that is row-sum normalized to $w$ produces the same total input as a local weight $w$:

$$w^{\text{fitted}} = \sum_j W_{ij}^{\text{ring}} \quad \text{(row-sum constraint)}$$

The single-node fit gives the target row-sum. The ring distributes that sum spatially according to a kernel. All resting-state firing rates, KO conditions, and bistable fixed points are preserved exactly.

---

## PYR → PYR (NMDA-gated, full Gaussian)

### Current model
- Local recurrence: $J_{\text{NMDA}}^{\text{local}} \cdot S_i^{\text{NMDA}}$ (self only, inside divisive PV term)
- Inter-node: $\sum_{j \neq i} W_{ij}^{\text{Gauss}} \cdot r_j^{\text{PYR}}$ (rate-based, separate parameter)

### Proposed model
All PYR→PYR recurrence is NMDA-gated and carried by a single Gaussian weight matrix **with non-zero diagonal**:

$$I_i^{\text{PYR,NMDA}} = \frac{\sum_j W_{ij}^{\text{PYR}} \cdot S_j^{\text{NMDA}}}{1 + g_{\text{GABA}} \cdot w_{pe}^{\text{fitted}} \cdot r_i^{\text{PV}}}$$

where:
- $W_{ij}^{\text{PYR}}$ is the Gaussian kernel **including the diagonal** ($W_{ii} \neq 0$)
- Row-sum normalized: $\sum_j W_{ij}^{\text{PYR}} = J_{\text{NMDA}}^{\text{fitted}}$ for all $i$
- $S_j^{\text{NMDA}}$ encodes node $j$'s activity (silent node → $S_j \approx 0$, active node → $S_j > 0$)
- The NMDA gating variable remains a vector of length $N$ (one per node, not $N^2$), since $S_{ji}$ depends only on the presynaptic node $j$: $S_{ji} \equiv S_j$

### Conversion
$$\sum_j W_{ij}^{\text{PYR}} = J_{\text{NMDA}}^{\text{fitted}}$$

The Gaussian shape ($\sigma_{\text{pyr}}$) determines the local/lateral split naturally — no partition parameter needed. The self-weight $W_{ii}$ corresponds to the Gaussian evaluated at distance 0, normalized within the full kernel.

### Why $S_j$ is sufficient (no explicit rate needed)
At steady state: $S_j^* = \frac{\gamma \tau_{\text{NMDA}} \cdot r_j^{\text{PYR}}}{1 + \gamma \tau_{\text{NMDA}} \cdot r_j^{\text{PYR}}}$. The rate of node $j$ is encoded monotonically in $S_j$ — no need for $r_j^{\text{PYR}}$ to appear separately.

---

## SOM → PYR (fully lateral)

### Current model
- Local: $w_{se} \cdot r_i^{\text{SOM}}$ (self-inhibition, subtractive)
- Inter-node: none

### Proposed model
SOM receives input from local PYR only, but projects **laterally** to neighboring nodes (zero self-connection). The local $w_{se}$ term is removed from the PYR equation and replaced by a lateral kernel:

$$I_{\text{SOM-lat},i}^{\text{PYR}} = \sum_{j \neq i} W_{ij}^{\text{SOM}} \cdot r_j^{\text{SOM}}$$

### Conversion
$$\sum_{j \neq i} W_{ij}^{\text{SOM}} = w_{se}^{\text{fitted}}$$

A uniform kernel (like current `w_pv_global`) is the simplest choice. A Gaussian surround is also possible.

### What changes away from homogeneity
At the bump center, the local model self-suppresses the active node; the lateral model suppresses neighbors instead. Bump-center nodes are partially released from SOM inhibition → bump amplitude may increase slightly. `w_pv_global` is the natural compensatory parameter.

---

## PV → PYR (fully divisive, local + lateral)

### Current model
- Local: divisive term $\frac{J_{\text{NMDA}} \cdot S}{1 + g_{\text{GABA}} \cdot w_{pe} \cdot r_i^{\text{PV}}}$
- Inter-node: **subtractive** $g_{\text{GABA}} \cdot w_{\text{PV}}^{\text{global}} \cdot \langle r^{\text{PV}} \rangle$ ← biological inconsistency

### Proposed model
All PV→PYR inhibition is divisive (perisomatic shunting regardless of whether PV is local or lateral). Local and inter-node PV contributions enter the same denominator:

$$I_i^{\text{PYR,NMDA}} = \frac{\sum_j W_{ij}^{\text{PYR}} \cdot S_j^{\text{NMDA}}}{1 + g_{\text{GABA}} \cdot \left( w_{pe}^{\text{local}} \cdot r_i^{\text{PV}} + \sum_{j \neq i} W_{ij}^{\text{PV}} \cdot r_j^{\text{PV}} \right)}$$

### Conversion
At the homogeneous fixed point:

$$w_{pe}^{\text{local}} + w_{\text{PV}}^{\text{inter}} = w_{pe}^{\text{fitted}}$$

The simplest maximally consistent choice: set $w_{pe}^{\text{local}} = 0$ (fully lateral PV, symmetric to SOM), so the entire fitted $w_{pe}^{\text{fitted}}$ transfers to the inter-node uniform kernel. This removes the asymmetry between local and lateral PV entirely.

---

## Summary Table

| Connection | Single-node | Ring | Kernel | Conversion constraint |
|---|---|---|---|---|
| PYR→PYR | $J_{\text{NMDA}}^{\text{fitted}}$ (local NMDA) | Gaussian incl. diagonal, NMDA-gated | Gaussian ($\sigma_{\text{pyr}}$) | Row-sum = $J_{\text{NMDA}}^{\text{fitted}}$ |
| SOM→PYR | $w_{se}^{\text{fitted}}$ (local, subtractive) | Lateral only, zero self | Uniform or Gaussian surround | Row-sum excl. self = $w_{se}^{\text{fitted}}$ |
| PV→PYR | $w_{pe}^{\text{fitted}}$ (local, divisive) | Fully divisive, local + lateral | Uniform (lateral) | $w_{pe}^{\text{local}} + w_{\text{PV}}^{\text{inter}} = w_{pe}^{\text{fitted}}$ |

**In all cases:** homogeneous fixed point firing rates are preserved exactly. No new free parameters are introduced by the conversion — the spatial distribution is determined by the kernel shape ($\sigma_{\text{pyr}}$, uniform) which is a structural choice, not a fitted quantity.

---

## New free parameters introduced

None from the conversion itself. The only structural choices are:
- $\sigma_{\text{pyr}}$: Gaussian width for PYR→PYR (already exists)
- Whether SOM lateral kernel is uniform or Gaussian (recommend uniform to start)
- Whether PV split is fully lateral ($w_{pe}^{\text{local}} = 0$) or partially local

---

## Implementation notes

```python
# PYR→PYR: Gaussian kernel with non-zero diagonal
W_pyr[i, j] = exp(-d(i,j)^2 / (2 * sigma_pyr^2))   # includes i==j (distance=0, weight=1)
W_pyr[i, :] *= J_NMDA_fitted / W_pyr[i, :].sum()    # row-sum normalize to J_NMDA_fitted

# Inter-node NMDA input (matrix-vector product, S is length-N vector)
I_nmda[i] = sum_j W_pyr[i,j] * S[j]                 # replaces both local J_NMDA*S_i and old inter-node term

# SOM lateral kernel (uniform, no self)
W_som[i, j] = w_se_fitted / (N - 1)   for j != i
W_som[i, i] = 0

# PV fully lateral (uniform, no self), fully divisive
W_pv[i, j] = w_pe_fitted / (N - 1)    for j != i
W_pv[i, i] = 0
# Denominator: 1 + g_GABA * (W_pv @ r_pv)[i]
```

---

*Written: 2026-04-28 — not yet implemented or tested.*