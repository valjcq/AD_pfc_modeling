#!/bin/bash

# Change to project root directory (parent of scripts)
# Works whether script is executed or sourced
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.." || exit 1

echo "Starting WHITE noise ring sweeps (all patterns)..."

echo "Ring WHITE - None pattern"
python scripts/ring_transient_sweep.py \
  --input_params params_bistable/last_opti/bistable_params.json \
  --noise_type white \
  --som_pattern none \
  --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --workers 16

echo "Ring WHITE - Uniform pattern"
python scripts/ring_transient_sweep.py \
  --input_params params_bistable/last_opti/bistable_params.json \
  --noise_type white \
  --som_pattern uniform \
  --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --workers 16

echo "Ring WHITE - Gaussian pattern (with sigma_som sweep)"
python scripts/ring_transient_sweep.py \
  --input_params params_bistable/last_opti/bistable_params.json \
  --noise_type white \
  --som_pattern gaussian \
  --amplitudes 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --sigmas 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 \
  --sigma_som_values 5 10 15 20 25 \
  --workers 16

echo "All WHITE noise ring sweeps completed!"
