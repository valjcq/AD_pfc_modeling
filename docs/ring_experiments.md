# Ring Attractor Experiments

This document describes maintained ring-attractor experiments and outputs.

## Active Commands

- `ring-run`
- `ring-study`
- `ring-diffusion`
- `ring-noise-floor`
- `ring-calibrate`
- `ring-asymmetry`
- `ring-burnin-stability`

## 1) `ring-run`

Single-condition simulation with full plotting outputs.

Typical usage:

```bash
python -m circuit_model ring-run --condition WT --amplitude 30 --w_pyr_pyr_inter 8 --w_pv_global 10
```

Primary outputs include dashboard, connectivity visualization, and bump metrics over time.

## 2) `ring-study`

Multi-condition and optionally multi-amplitude batch study.

Typical usage:

```bash
python -m circuit_model ring-study --conditions WT WT_APP --n_trials 100 --amplitudes 20 30 --w_pyr_pyr_inter 8 --w_pv_global 10
```

Produces comparison plots across delay-time metrics and across amplitudes.

## 3) `ring-diffusion`

Mean-squared-displacement (MSD) analysis of bump-center diffusion during delay.

Typical usage:

```bash
python -m circuit_model ring-diffusion --conditions WT WT_APP --n_trials 50 --w_pyr_pyr_inter 8 --w_pv_global 10
```

Outputs include condition-level diffusion metrics and trajectory summaries.

## 4) `ring-noise-floor`

No-stimulus baseline trials to estimate amplitude threshold used to separate spontaneous activity from stable memory bumps.

Typical usage:

```bash
python -m circuit_model ring-noise-floor --conditions WT --w_inter_values 4 6 8 --w_pv_global 10
```

## 5) `ring-calibrate`

2D calibration sweep over cue amplitude and inter-node excitation (`w_pyr_pyr_inter`) using the noise-floor criterion.

Typical usage:

```bash
python -m circuit_model ring-calibrate --conditions WT WT_APP --amplitudes 20 30 40 --w_inter_values 6 8 10 --w_pv_global 10
```

Produces heatmaps, timecourse summaries, and recommended parameter points.

## 6) `ring-asymmetry`

Trial-wise pre-cue and delay asymmetry analysis with statistical reports.

Typical usage:

```bash
python -m circuit_model ring-asymmetry --conditions WT WT_APP a7_KO_APP --n_trials 100 --w_pyr_pyr_inter 8 --w_pv_global 10
```

Outputs include asymmetry distributions, pre-vs-delay correlation, and significance tables.

## 7) `ring-burnin-stability`

Stationarity check across burn-in windows using non-parametric tests.

Typical usage:

```bash
python -m circuit_model ring-burnin-stability --conditions WT --n_trials 100 --burnin_ms 10000 --period_ms 1000 --w_pyr_pyr_inter 8 --w_pv_global 10
```

Reports whether amplitude and asymmetry distributions remain stable across burn-in windows.
