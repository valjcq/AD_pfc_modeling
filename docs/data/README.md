# AD_data

Two-photon calcium imaging recordings from mouse prefrontal cortex (PFC). Data captures spontaneous spike activity (transients) across different genotypes in an Alzheimer's disease (AD) mouse model context.

## Experimental Design

**Disease model:** APP transgenic mice (amyloid precursor protein overexpression) vs. wild-type controls.

**Interneuron targeting:** Cre-driver lines allow cell-type-specific manipulation:
- **PV-Cre** — Parvalbumin interneurons (fast-spiking, perisomatic inhibition)
- **SST-Cre** — Somatostatin interneurons (dendritic inhibition)
- **VIP-Cre** — VIP interneurons (disinhibitory)

**Nicotinic receptor knockouts:** Effects of nicotinic acetylcholine receptors on network activity:
- **α7 KO** (`a7KO`) — Alpha-7 subunit knockout
- **β2 KO** (`b2KO`) — Beta-2 subunit knockout
- **α5 KO** (`a5KO`) — Alpha-5 subunit knockout
- **α7β2 KO** (`a7b2KO`) — Double knockout (α7 + β2)
- **Re-expression** (`_reexp`) — The knocked-out receptor is genetically re-expressed in the APP background; tests whether restoring the receptor rescues the APP-induced activity changes

**Timepoints post-injection:**
- `1mo_post_injection` — largest cohort, full genotype matrix
- `3mo_post_injection`
- `6mo_post_injection`

**Separate cohort:** `galantamine_2021` — pharmacological intervention (groups 1–3, ~5 mice each).

## Directory Structure

```
AD_data/
├── 1mo_post_injection/        # Main cohort
│   ├── WT/                    # Wild-type control
│   ├── WT_APP/                # AD model baseline
│   ├── WT_APP_reexp/
│   ├── PV_control/
│   ├── PV_APP/
│   ├── SST_control/
│   ├── SST_APP/
│   ├── VIP_control/
│   ├── VIP_APP/
│   ├── a5KO_control/ a5KO_APP/
│   ├── a7KO_control/ a7KO_APP/ a7KO_APP_reexp/
│   ├── b2KO_control/ b2KO_APP/ b2KO_APP_reexp/
│   └── a7b2KO_control/ a7b2KO_APP/
├── 3mo_post_injection/        # Subset of genotypes (WT, WT_APP, a7KO, b2KO)
├── 6mo_post_injection/        # Subset of genotypes (WT, WT_APP, a7KO)
├── galantamine_2021/          # Drug cohort, group_1–group_3
├── additional/                # Supplementary cohort
└── parameters.txt             # Root-level time constant
```

Each group contains numbered mouse subfolders (`mouse_1`, `mouse_2`, …).

## File Formats

### `Results{N}.csv`
One file per recording session/field of view. Columns `Mean1`–`Mean10` (or more) contain fluorescence time series for individual neurons (ROIs). Values are in raw fluorescence units.

Multiple Results files per mouse = multiple sessions or fields of view.

### `parameters.rtf` / `parameters.txt`
Single value `t = <value> sec` — total recording duration in seconds (~164–165 s typical).

## Computed Activity Rates

Mean activity rates per genotype are pre-computed in `AD_data/summary/` using
`scripts/compute_target_rates.py`. Formula: `mean(all fluorescence values) / t_recording`.

| File | Contents |
|---|---|
| `summary/targets_1mo.json` | All 1mo genotypes (control + APP + reexp + double KO) |
| `summary/targets_3mo.json` | 3mo genotypes |
| `summary/targets_6mo.json` | 6mo genotypes |
| `summary/targets_all.json` | All timepoints merged |

Each entry contains `mean`, `median`, `std`, `n_files`, `n_mice`, and `per_mouse` rates.

### Key 1mo rates (mean, in fluorescence units / s)

| Genotype | mean rate | median | n files |
|---|:---:|:---:|:---:|
| WT               | 4.143 | 3.785 | 37 |
| WT_APP           | 4.566 | 4.394 | 33 |
| WT_APP_reexp     | 3.930 | 3.813 | 18 |
| PV_control       | 2.079 | 1.602 | 23 |
| PV_APP           | 3.198 | 3.017 | 34 |
| SST_control      | 3.423 | 2.976 | 26 |
| SST_APP          | 3.577 | 3.454 | 50 |
| VIP_control      | 1.933 | 1.857 | 27 |
| VIP_APP          | 1.633 | 1.363 | 39 |
| a7KO_control     | 3.513 | 3.387 | 21 |
| a7KO_APP         | 4.270 | 4.085 | 38 |
| a7KO_APP_reexp   | 3.147 | 2.683 | 20 |
| a7b2KO_control   | 4.256 | 4.195 | 20 |
| a7b2KO_APP       | 3.138 | 2.691 | 18 |
| b2KO_control     | 4.800 | 4.852 | 33 |
| b2KO_APP         | 4.123 | 3.765 | 28 |
| b2KO_APP_reexp   | 3.545 | 3.282 | 17 |
| a5KO_control     | 3.790 | 3.722 | 44 |
| a5KO_APP         | 3.491 | 3.511 | 12 |

See `docs/data/fitting_roadmap.md` for how these rates map to circuit model fit targets.
