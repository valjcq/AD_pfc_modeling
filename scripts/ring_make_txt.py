"""
Regenerate a .txt fit summary for an existing ring-optimize JSON output.

Usage
-----
python scripts/ring_make_txt.py \
    --circuit_json params/new/ring_firing_rate/WT_1mo_article_ko.json \
    --ring_json    params/new/ring_firing_rate/WT_1mo_article_ko_ring.json \
    --target_pyr 8.214 --target_som 4.295 --target_pv 4.073 --target_vip 6.051

Optional KO targets (omit if not fitted):
    --target_alpha7_ko_pyr 17.539
    --target_beta2_ko_pyr  17.965
    --target_alpha5_ko_pyr 9.285

Optional simulation settings (defaults match ring-optimize defaults):
    --n_trials 3  --T_ms 2500  --dt_ms 0.1  --burn_in_ms 500  --window_ms 2000
    --n_nodes 64

If _fit_metadata is already present in the JSON, it is used directly without
re-running any simulation (pass --force to re-run anyway).
"""
import argparse
import json
import sys
from pathlib import Path

# Make sure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuit_model.io import (
    load_params_json, save_params_json, save_fit_summary_txt, build_fit_comparison,
)
from circuit_model.jacobian import compute_jacobian
from circuit_model.loss import TargetRates, FitConfig
from circuit_model.optimization import KOMeans, run_condition, _build_conditions
from circuit_model.ring.params import RingParams
from circuit_model.ring.optimization import RingFitConfig, run_ring_trials


def _load_ring_params(path: str, n_nodes: int) -> RingParams:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return RingParams(
        n_nodes=d.get("n_nodes", n_nodes),
        w_pyr_pyr_inter=d["w_pyr_pyr_inter"],
        w_pv_global=d["w_pv_global"],
        sigma_pyr_deg=d["sigma_pyr_deg"],
    )


def main():
    parser = argparse.ArgumentParser(description="Regenerate .txt summary for a ring-optimize result.")
    parser.add_argument("--circuit_json", required=True)
    parser.add_argument("--ring_json",    required=True)
    parser.add_argument("--target_pyr",   type=float, required=True)
    parser.add_argument("--target_som",   type=float, required=True)
    parser.add_argument("--target_pv",    type=float, required=True)
    parser.add_argument("--target_vip",   type=float, required=True)
    parser.add_argument("--target_alpha7_ko_pyr", type=float, default=None)
    parser.add_argument("--target_beta2_ko_pyr",  type=float, default=None)
    parser.add_argument("--target_alpha5_ko_pyr", type=float, default=None)
    parser.add_argument("--n_trials",  type=int,   default=3)
    parser.add_argument("--T_ms",      type=float, default=2500.0)
    parser.add_argument("--dt_ms",     type=float, default=0.1)
    parser.add_argument("--burn_in_ms",type=float, default=500.0)
    parser.add_argument("--window_ms", type=float, default=2000.0)
    parser.add_argument("--n_nodes",   type=int,   default=64)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--force",     action="store_true",
                        help="Re-run simulation even if _fit_metadata already present in JSON")
    args = parser.parse_args()

    # --- Load params ---
    circuit_params = load_params_json(args.circuit_json)
    ring_params    = _load_ring_params(args.ring_json, args.n_nodes)

    # --- Check for existing metadata ---
    with open(args.circuit_json, encoding="utf-8") as f:
        raw = json.load(f)
    existing_meta = raw.get("_fit_metadata")

    if existing_meta and not args.force:
        print("Found _fit_metadata in JSON — using stored values (pass --force to re-run).")
        fit_meta = existing_meta
        save_fit_summary_txt(args.circuit_json, fit_meta, params=circuit_params)
        txt_path = str(Path(args.circuit_json).with_suffix(".txt"))
        print(f"Written: {txt_path}")
        return

    # --- Re-run evaluation ---
    import numpy as np

    target = TargetRates(
        mean_r_pyr=args.target_pyr,
        mean_r_som=args.target_som,
        mean_r_pv=args.target_pv,
        mean_r_vip=args.target_vip,
        alpha7_ko_pyr=args.target_alpha7_ko_pyr,
        beta2_ko_pyr=args.target_beta2_ko_pyr,
        alpha5_ko_pyr=args.target_alpha5_ko_pyr,
    )
    fit_cfg = FitConfig(
        T_ms=args.T_ms, dt_ms=args.dt_ms, burn_in_ms=args.burn_in_ms,
        window_ms=args.window_ms, n_trials=args.n_trials,
    )
    ring_cfg = RingFitConfig(fit_cfg=fit_cfg, n_trials_ring=args.n_trials)
    rng = np.random.default_rng(args.seed)

    print(f"Running {args.n_trials} ring trial(s) at rest (n_nodes={ring_params.n_nodes})...")
    ok, ring_means = run_ring_trials(circuit_params, ring_params, ring_cfg, rng)
    if not ok:
        print("WARNING: simulation diverged — means may be unreliable.")

    print("Running KO conditions (single-node)...")
    conditions = _build_conditions(circuit_params, target, fit_cfg, rng)
    ko_means = KOMeans()
    ko_attr = {"alpha7_ko": "alpha7_ko", "alpha5_ko": "alpha5_ko", "beta2_ko": "beta2_ko"}
    for cond in conditions:
        cond_name, ok, means_arr = run_condition(cond)
        if cond_name in ko_attr:
            setattr(ko_means, ko_attr[cond_name], means_arr)

    J = compute_jacobian(circuit_params, ring_means)

    fit_meta = build_fit_comparison(ring_means, ko_means, target, float("nan"), jacobian=J)
    fit_meta["ring_params"] = {
        "w_pyr_pyr_inter": round(float(ring_params.w_pyr_pyr_inter), 6),
        "w_pv_global":     round(float(ring_params.w_pv_global), 6),
        "sigma_pyr_deg":   round(float(ring_params.sigma_pyr_deg), 6),
        "n_nodes":         int(ring_params.n_nodes),
    }

    save_params_json(args.circuit_json, circuit_params, fit_meta=fit_meta)
    save_fit_summary_txt(args.circuit_json, fit_meta, params=circuit_params)
    txt_path = str(Path(args.circuit_json).with_suffix(".txt"))
    print(f"Written: {txt_path}")


if __name__ == "__main__":
    main()
