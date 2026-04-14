# Best-Fit Parameter Backups

This directory stores reference parameter sets to avoid accidental overwriting.

## Files (Ranked by Recommendation)

### ⭐ **RECOMMENDED FOR ALL OPTIMIZATION RUNS**
- **current_schema_reference_20260413.json**  
  Current-schema reference uses `J_NMDA` (not deprecated `w_ee`)  
  **Verified:** All 25 parameters within bounds (2026-04-13 verification)  
  **Status:** Production-ready ✓  
  **Use for:** `--params_json params/best_fit_params/current_schema_reference_20260413.json`

### Legacy (Requires Auto-Migration)
- **WT_1mo_circuit_reference_20260413.json**  
  Old-schema WT from ring_optimize/A1/ (uses deprecated `w_ee`)  
  **Migration:** Auto-converts `w_ee → J_NMDA` on load (circuit_model/io.py:50)  
  ⚠️ **Not recommended** — use current_schema_reference instead  
  **Status:** Still functional, but generates deprecation warning

- **bistable_attempt_20260413.json**  
  Result from early bistable optimization run  
  **Use for:** Archive/reference only

## How They're Used

### Optimization Modes

#### Standard Mode (Recommended for General Use)
Optimizes firing rates for all four cell types simultaneously.

```bash
python -m circuit_model optimize \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --n_samples 20000 \
  --params_json params/best_fit_params/current_schema_reference_20260413.json \
  --output_dir standard_optimize_long/
```

#### Bistable Mode (Recommended for Bistability Research)
Optimizes firing rates with bistability constraints between low and high states.

```bash
python -m circuit_model optimize \
  --mode bistable \
  --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051 \
  --r_low_hz 8.214 --r_high_hz 30.0 --r_mid_hz 15.0 \
  --w_bistab 1.0 --w_rate_bistab 0.0 --w_margin 0.0 --w_physiol 1.0 \
  --budget 20000 \
  --params_json params/best_fit_params/current_schema_reference_20260413.json \
  --noise_type none \
  --n_trials 1 \
  --output_dir bistable_physiol/
```

| Parameter | Standard Mode | Bistable Mode |
|-----------|---------------|---------------|
| `--mode` | default (omitted) | `bistable` |
| `--budget` | `--n_samples 20000` | `--budget 20000` |
| `--noise_type` | default | `none` |
| `--n_trials` | default | `1` |


### Save Results
Results are automatically saved to `best_params.json` in the working directory. To preserve them, copy to this directory with a timestamp:
```bash
cp best_params.json params/best_fit_params/optimization_result_$(date +%Y%m%d_%H%M%S).json
```

## Important

- `best_params.json` in the project root is **overwritten by each optimization run**
- Always keep backups in this directory with descriptive timestamps
- The WT reference should not be modified directly
