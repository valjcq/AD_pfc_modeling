# CLI Reference

The unified CLI is invoked via:

```bash
python -m circuit_model <command>
```

Available commands:

```bash
python -m circuit_model {run,optimize,study,ring-run,ring-study,ring-diffusion,ring-noise-floor,ring-calibrate,ring-asymmetry,ring-burnin-stability} [options]
```

## Commands

1. `run`: single-circuit simulation and plots.
2. `optimize`: Nevergrad parameter optimization against target rates.
3. `study`: batch single-circuit study across 8 receptor conditions.
4. `ring-run`: ring attractor simulation for one condition.
5. `ring-study`: ring attractor multi-condition and multi-amplitude comparison.
6. `ring-diffusion`: bump diffusion/MSD analysis during delay.
7. `ring-noise-floor`: baseline no-stimulus runs to estimate noise-floor threshold.
8. `ring-calibrate`: 2D sweep over cue amplitude and `w_pyr_pyr_inter`.
9. `ring-asymmetry`: trial-wise left/right asymmetry analysis.
10. `ring-burnin-stability`: stationarity check over burn-in windows.

## Notes

- Removed commands are no longer part of the supported interface.
- Ring commands share common options for ring size, coupling, delay, and noise.
- Use `python -m circuit_model <command> --help` for full argument details.
