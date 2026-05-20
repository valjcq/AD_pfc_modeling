# Single-Node to Ring Weight Conversion

**Status:** Implemented in `circuit_model/ring/connectivity.py`. See [ring_attractor.md §2](ring_attractor.md#2-inter-node-connectivity) for the live description; this file is kept as design documentation showing how the principle was derived.

The core idea: the single-node optimization produces fitted weights that represent the *total* synaptic drive onto each population. When moving to the ring network, this total drive is preserved at the homogeneous fixed point by distributing it spatially according to a connectivity kernel. Fixed-point equivalence is guaranteed by construction via row-sum normalization.

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

## SOM → PYR (configurable lateral pattern)

### Current model
- Local: $w_{se} \cdot r_i^{\text{SOM}}$ (self-inhibition, subtractive)
- Inter-node: none

### Implemented model
SOM still projects subtractively to PYR, but the kernel is now selectable via `RingParams.som_pattern`. All variants are row-sum normalised to $w_{se}^{\text{fitted}}$:

- **`gaussian` (default)** — annular Gaussian centred at distance $\mu = 3\,\sigma_{\text{pyr}}$ from each source, width $\sigma_{\text{som}}$, zero diagonal. Lateral surround that recovers the Mexican-hat motif.
- **`uniform`** — uniform inhibition everywhere except a Gaussian hole of half-width $2\,\sigma_{\text{som}}$ around each source, zero diagonal.
- **`none`** — diagonal matrix with $w_{se}$ on every entry; reproduces the original single-node SOM term node-by-node, no lateral coupling.

For the two lateral patterns, $\sum_{j \neq i} W_{ij}^{\text{SOM}} = w_{se}^{\text{fitted}}$. For `none`, the diagonal carries the full $w_{se}^{\text{fitted}}$.

### What changes away from homogeneity
At the bump center, the `none` pattern self-suppresses the active node; the lateral patterns suppress neighbours instead. Bump-center nodes are then partially released from SOM inhibition, which can broaden the bump and modify its amplitude — `sigma_som_deg` is the main shape parameter for tuning this trade-off.

---

## PV → PYR (fully divisive, uniform all-to-all)

### Current model
- Local: divisive term $\frac{J_{\text{NMDA}} \cdot S}{1 + g_{\text{GABA}} \cdot w_{pe} \cdot r_i^{\text{PV}}}$
- Inter-node: **subtractive** $g_{\text{GABA}} \cdot w_{\text{PV}}^{\text{global}} \cdot \langle r^{\text{PV}} \rangle$ ← biological inconsistency

### Implemented model
All PV→PYR inhibition is divisive. The PV kernel is a single uniform matrix with $W^{\text{PV}}_{ij} = w_{pe}/N$ for every pair (including the diagonal), and the full row-sum $w_{pe}$ enters the denominator:

$$I_i^{\text{PYR,NMDA}} = \frac{\sum_j W_{ij}^{\text{PYR}} \cdot S_j^{\text{NMDA}}}{1 + g_{\text{GABA}} \cdot \sum_j W_{ij}^{\text{PV}} \cdot r_j^{\text{PV}}}$$

### Conversion
At the homogeneous fixed point $\sum_j W^{\text{PV}}_{ij} r_j^{\text{PV}} = w_{pe} \cdot r^{\text{PV}}$, recovering the single-node denominator exactly. There is no separate local/lateral split — the uniform kernel handles both contributions in one matrix.

---

## Summary Table

| Connection | Single-node | Ring | Kernel | Conversion constraint |
|---|---|---|---|---|
| PYR→PYR | $J_{\text{NMDA}}^{\text{fitted}}$ (local NMDA) | Gaussian incl. diagonal, NMDA-gated | Gaussian ($\sigma_{\text{pyr}}$) | Row-sum = $J_{\text{NMDA}}^{\text{fitted}}$ |
| SOM→PYR | $w_{se}^{\text{fitted}}$ (local, subtractive) | Configurable (`gaussian` / `uniform` / `none`) | Annular Gaussian, flat-with-hole, or diagonal | Row-sum = $w_{se}^{\text{fitted}}$ |
| PV→PYR | $w_{pe}^{\text{fitted}}$ (local, divisive) | Fully divisive, uniform all-to-all | Uniform incl. diagonal | Row-sum = $w_{pe}^{\text{fitted}}$ |

**In all cases:** homogeneous fixed point firing rates are preserved exactly. No new free parameters are introduced by the conversion — the spatial distribution is determined by the kernel shape ($\sigma_{\text{pyr}}$, uniform) which is a structural choice, not a fitted quantity.

---

## New free parameters introduced

None from the conversion itself. The structural choices currently exposed are:
- $\sigma_{\text{pyr}}$: Gaussian width for PYR→PYR
- $\sigma_{\text{som}}$ and `som_pattern`: SOM→PYR shape (annular Gaussian, flat-with-hole, or local-only)

PV→PYR is fixed to uniform all-to-all and exposes no shape parameter.

---

## Implementation reference

```python
# PYR→PYR: Gaussian kernel with non-zero diagonal
W_pyr[i, j] = exp(-d(i,j)^2 / (2 * sigma_pyr^2))   # includes i==j (distance=0, weight=1)
W_pyr[i, :] *= J_NMDA_fitted / W_pyr[i, :].sum()    # row-sum normalize to J_NMDA_fitted

# NMDA input (matrix-vector product, S is length-N vector)
I_nmda[i] = sum_j W_pyr[i,j] * S[j]

# SOM "gaussian" pattern: annular surround, zero diagonal
mu = 3.0 * sigma_pyr_rad
W_som[i, j] = exp(-(d(i,j) - mu)^2 / (2 * sigma_som^2))   for j != i
W_som[i, i] = 0
W_som[i, :] *= w_se_fitted / W_som[i, :].sum()

# PV: uniform all-to-all including diagonal, fully divisive
W_pv[i, j] = w_pe_fitted / N                          # for all i, j (incl. diagonal)
# Denominator: 1 + g_GABA * (W_pv @ r_pv)[i]
```

See `circuit_model/ring/connectivity.py` for the live implementation, including the `uniform` and `none` SOM variants.