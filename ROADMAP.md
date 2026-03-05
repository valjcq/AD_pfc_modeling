# Roadmap — Refaire les figures (post bugfix burn-in)

**Bug corrigé :** bruit dans le burn-in (`c8bdaf8`). Tout est à relancer.

**Légende :**
- `[ ]` = à faire
- `[x]` = fait / validé
- `[?]` = nécessite une décision après résultats précédents

---

## CONFIG A — Inhibition faible (`--w_pv_global 4`)

> L'excitation (`--w_pyr_pyr_inter`) est *sweepée* dans noise-floor et calibrate.
> Pour study/diffusion/distractor, remplacer `W_INTER_A` par la valeur choisie après calibration.

---

### A1 · Noise Floor

```bash
python3 -m circuit_model ring-noise-floor \
  --w_pv_global 4 --w_pyr_pyr_inter 4.0 \
  --w_inter_values 2.0 3.0 4.0 5.0 5.5 6.0 7.0 \
  --n_baseline 300 \
  --conditions WT WT_APP a7_KO_APP \
  --no_show
```

- [x] Commande lancée
- [x] Résultats vérifiés → `figs/ring/calibration/128_inhib_4/`

---

### A2 · Calibration

> Dépend de A1 (noise floor utilisé automatiquement).

```bash
python3 -m circuit_model ring-calibrate \
  --w_pv_global 4 --w_pyr_pyr_inter 4.0 \
  --w_inter_values 2.0 3.0 4.0 5.0 5.5 6.0 7.0 \
  --amplitudes 10 15 20 25 30 35 40 45 \
  --conditions WT WT_APP a7_KO_APP \
  --no_show
```

- [x] Commande lancée
- [x] Heatmap de calibration vérifiée → choisir `W_INTER_A` optimal
- [x] `W_INTER_A` = 5.5 ou 5.0 (difference de noise floor entre WT et WT_APP pour 5.5, pas pour 5.0)

---

### A3 · Study (WT vs WT_APP)

> Remplacer `W_INTER_A` par la valeur choisie en A2.

```bash
python3 -m circuit_model ring-study \
  --w_pv_global 4 --w_pyr_pyr_inter 5.0 \
  --amplitudes 10 15 20 25 30 35 40 45 \
  --conditions WT WT_APP \
  --delay_ms 5000 \
  --no_show
```

- [x] `W_INTER_A` renseigné
- [x] Commande lancée
- [x] Figures vérifiées → `figs/ring/run/128_inhib_4_excit_5.0/`

---

### A4 · Diffusion (amp = 30)

```bash
python3 -m circuit_model ring-diffusion \
  --w_pv_global 4 --w_pyr_pyr_inter 5.0 \
  --amplitude 30 \
  --conditions WT WT_APP a7_KO_APP \
  --no_show
```

- [x] `W_INTER_A` renseigné
- [x] Commande lancée
- [x] Figures vérifiées → `figs/ring/diffusion/128_inhib_4_excit_5.0/amp30/`

---

### A5 · Distractor Sweep (WT)

> Amplitude du cue : choisir une amplitude qui maintient le bump selon la calibration.
> **À déterminer après A2.** Remplacer `AMP_A` (suggestion : 30 ou 45).

```bash
python3 -m circuit_model ring-distractor-sweep \
  --w_pv_global 4 --w_pyr_pyr_inter 5.0 \
  --amplitude 30 \
  --condition WT \
  --no_show
```

- [x] `W_INTER_A` et `AMP_A` renseignés
- [x] Commande lancée
- [x] Figures vérifiées → `figs/ring/distractor_sweep/default/128_inhib_4_excit_{W_INTER_A}/WT/amp{AMP_A}/`

---
---

## CONFIG B — Inhibition forte (`--w_pv_global 10`)

> Valeur cible d'excitation : `--w_pyr_pyr_inter 7` (à confirmer via calibration).
> Pour study/diffusion/run/asymmetry, remplacer `W_INTER_B` (probablement 7).

---

### B1 · Noise Floor

```bash
python3 -m circuit_model ring-noise-floor \
  --w_pv_global 10 --w_pyr_pyr_inter 7.0 \
  --w_inter_values 2.0 4.0 6.0 6.25 6.5 6.75 7.0 7.25 7.5 7.75 8.0 8.25 8.5 8.75 9.0 9.5 10.0 10.5 11.0 11.5 12.0 \
  --n_baseline 300 \
  --conditions WT WT_APP \
  --no_show
```

- [x] Commande lancée
- [x] Résultats vérifiés → `figs/ring/calibration/128_inhib_10/`

---

### B2 · Calibration

> Dépend de B1. Amplitudes : 10–50 (note : 50 en plus vs Config A).

```bash
python3 -m circuit_model ring-calibrate \
  --w_pv_global 10 --w_pyr_pyr_inter 7.0 \
  --w_inter_values 7.0 8.0 8.25 8.5 8.75 9.0 8.5 10.0 11.0 \
  --amplitudes 10 15 20 25 30 40 45 50 \
  --conditions WT WT_APP a7_KO_APP \
  --no_show
```

- [x] Commande lancée
- [x] Heatmap vérifiée → confirmer `W_INTER_B`
- [x] `W_INTER_B` = 8.5 ou 8.0

---

### B3 · Study (WT vs WT_APP)

```bash
python3 -m circuit_model ring-study \
  --w_pv_global 10 --w_pyr_pyr_inter 8.0 \
  --amplitudes 20 25 30 35 40 45 \
  --conditions WT WT_APP \
  --delay_ms 5000 \
  --no_show
```

- [x] `W_INTER_B` renseigné
- [x] Commande lancée
- [x] Figures vérifiées → `figs/ring/run/128_inhib_10_excit_8.0/`

---

### B4 · Diffusion (amp = 20)

```bash
python3 -m circuit_model ring-diffusion \
  --w_pv_global 10 --w_pyr_pyr_inter 8.0 \
  --amplitude 20 \
  --conditions WT WT_APP a7_KO_APP \
  --no_show
```

- [x] `W_INTER_B` renseigné
- [x] Commande lancée
- [x] Figures vérifiées → `figs/ring/diffusion/128_inhib_10_excit_8.0/amp20/`

---

### B5 · Ring Study amp = 20 (WT vs WT_APP)

> Equivalant à B3 mais à amplitude fixe amp=40 pour comparaison directe.

```bash
python3 -m circuit_model ring-study \
  --w_pv_global 10 --w_pyr_pyr_inter 8.0 \
  --amplitudes 20 \
  --conditions WT WT_APP \
  --delay_ms 5000 \
  --no_show
```

- [x] `W_INTER_B` renseigné
- [ ] Commande lancée

---

### B6 · Asymmetry (WT, WT_APP, a7_KO_APP)

```bash
python3 -m circuit_model ring-asymmetry \
  --w_pv_global 10 --w_pyr_pyr_inter 8.0 \
  --amplitude 20 \
  --conditions WT WT_APP a7_KO_APP \
  --no_show
```

- [ ] `W_INTER_B` renseigné
- [ ] Commande lancée
- [ ] Figures vérifiées → `figs/ring/asymmetry/128_inhib_10_excit_{W_INTER_B}/amp40_corrected/`


---

## Ordre d'exécution recommandé

```
A1 (noise floor)  →  A2 (calibrate, choisir W_INTER_A)
B1 (noise floor)  →  B2 (calibrate, confirmer W_INTER_B)

Ensuite en parallèle :
  A3, A4, A5 (avec W_INTER_A)
  B3, B4, B5, B6 (avec W_INTER_B)
```

> **Note :** A1 et B1 peuvent être lancés simultanément (configs indépendantes).
> A2 et B2 peuvent aussi tourner en parallèle après leurs noise floors respectifs.

---

## Paramètres fixes (tous les runs)

| Paramètre | Valeur |
|-----------|--------|
| `--n_nodes` | 128 (défaut) |
| `--delay_ms` | 5000 |
| `--sigma_pyr_deg` | 30.0 (défaut) |

---

## Outputs attendus

| Commande | Dossier de sortie |
|----------|-------------------|
| ring-noise-floor | `figs/ring/calibration/128_inhib_{pv}/` |
| ring-calibrate | `figs/ring/calibration/128_inhib_{pv}/` |
| ring-study / ring-run | `figs/ring/run/128_inhib_{pv}_excit_{w}/` |
| ring-diffusion | `figs/ring/diffusion/128_inhib_{pv}_excit_{w}/amp{a}/` |
| ring-distractor-sweep | `figs/ring/distractor_sweep/default/128_inhib_{pv}_excit_{w}/{cond}/amp{a}/` |
| ring-asymmetry | `figs/ring/asymmetry/128_inhib_{pv}_excit_{w}/amp{a}_corrected/` |
