#!/bin/bash
# APP condition simulated via nAChR blockade (Koukouli et al. 2025):
#   α7 −90%, α5 −40%, β2 −12.5% activation applied on top of bistable_params.json

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || exit 1

PARAMS="params_bistable/last_opti/bistable_params.json"
OUTBASE="params_bistable/last_opti"

echo "=== APP sim sweeps (OU noise only) ==="

echo "Ring OU - None pattern (APP sim)"
python scripts/ring_transient_sweep.py \
  --input_params "$PARAMS" \
  --noise_type ou \
  --som_pattern none \
  --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --condition APP_sim \
  --output_dir "$OUTBASE/ring_transient_sweep_som_none_app" \
  --workers 16

echo "Ring OU - Gaussian pattern (APP sim, sigma_som sweep)"
python scripts/ring_transient_sweep.py \
  --input_params "$PARAMS" \
  --noise_type ou \
  --som_pattern gaussian \
  --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --sigma_som_values 5 10 15 20 25 \
  --condition APP_sim \
  --output_dir "$OUTBASE/ring_transient_sweep_app" \
  --workers 16

echo "Bistable OU noise (APP sim)"
python scripts/bistable_transient_sweep.py \
  --input_params "$PARAMS" \
  --noise_type ou \
  --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --condition APP_sim \
  --output_dir "$OUTBASE/transient_sweep_app" \
  --workers 16

echo "=== All APP sim sweeps completed ==="
