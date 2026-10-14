"""Local robustness check around a fitted bistable parameter set.

Tests, all scored with the same bistable_loss used during the fit:
  1. 1-D profiles: scale each free parameter by factors in [0.5, 2] (others fixed).
  2. Joint perturbations: multiply free parameters by exp(N(0, s)) for several s,
     either all at once or per group (weights / tonic drives / nAChR currents).
For each candidate we record the loss, whether the circuit is still bistable, and the
PYR/SOM/PV/VIP rates at the low and high fixed points.

Usage:
    PYTHONPATH=. python scripts/robustness_check.py params_bistable/last_opti1 \
        --out params_bistable/last_opti1/robustness
"""
import argparse
import dataclasses
import json
import shlex
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from circuit_model.bistable_loss import BistableConfig, bistable_loss
from circuit_model.io import load_params_json
from circuit_model.params import CircuitParams, default_bounds

FACTORS = np.geomspace(0.5, 2.0, 13)
SIGMAS = [0.05, 0.10, 0.20, 0.30]
GROUPS = {
    "all": lambda n: True,
    "weights": lambda n: n.startswith(("w_", "J_NMDA", "g_")),
    "tonic_drives": lambda n: n.startswith("I0_"),
    "nAChR_currents": lambda n: n.startswith("I_"),
    "adaptation": lambda n: "adapt" in n,
}


def _cli_args(fit_dir: Path) -> dict[str, str]:
    """Parse '--key value' pairs from the fit's command.txt."""
    toks = shlex.split((fit_dir / "command.txt").read_text())
    return {t[2:]: toks[i + 1] for i, t in enumerate(toks)
            if t.startswith("--") and i + 1 < len(toks) and not toks[i + 1].startswith("--")}


def build_config(a: dict[str, str]) -> BistableConfig:
    """Mirror the BistableConfig built by `optimize --mode bistable` in cli.py."""
    f = lambda k, d: float(a.get(k, d))
    return BistableConfig(
        r_low_target=f("r_low_hz", a["target_pyr"]),
        delta_r_min=f("delta_r_min", 15.0),
        r_pv_target=f("target_pv", 0), r_som_target=f("target_som", 0), r_vip_target=f("target_vip", 0),
        r_pyr_high_target=f("r_pyr_high_target", 60.2), r_som_high_target=f("r_som_high_target", 35.2),
        r_pv_high_target=f("r_pv_high_target", 35.3), r_vip_high_target=f("r_vip_high_target", 68.8),
        w_bistab=f("w_bistab", 5.0), w_rate=f("w_rate_bistab", 1.0), w_rate_high=f("w_rate_high", 1.5),
        w_margin=f("w_margin", 2.0), w_jacobian=f("jacobian_weight", 1.0),
        nullcline_peak_max=f("nullcline_peak_max", 80.0), w_peak=f("w_peak", 0.0),
    )


RATE_KEYS = {  # output name -> bistable_loss component
    "low_pyr": "r_low_fp", "low_som": "r_som_fp", "low_pv": "r_pv_fp", "low_vip": "r_vip_fp",
    "high_pyr": "r_high_fp", "high_som": "r_som_high_fp", "high_pv": "r_pv_high_fp", "high_vip": "r_vip_high_fp",
}


def _eval(job):
    d, cfg = job
    try:
        L, c = bistable_loss(CircuitParams(**d), cfg, return_components=True)
    except Exception:
        return float("nan"), False, {}
    bistable = bool(c.get("n_stable", 0) >= 2 and c.get("L_bistab", 1) == 0)
    rates = {k: float(c[v]) for k, v in RATE_KEYS.items() if v in c and c[v] is not None}
    return float(L), bistable, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_joint", type=int, default=300)
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()

    fit_dir = Path(args.fit_dir)
    a = _cli_args(fit_dir)
    cfg = build_config(a)
    base = load_params_json(str(fit_dir / "best_params.json"))
    bounds = default_bounds(base, w_hi=float(a["w_hi"]) if "w_hi" in a else None)
    freeze = set(a.get("freeze", "").split(","))
    base_d = dataclasses.asdict(base)
    # Skip frozen params and params at exactly 0 (multiplicative perturbation is a no-op).
    free = [n for n in bounds if n not in freeze and base_d[n] != 0]

    rng = np.random.default_rng(0)
    jobs, tags = [(base_d, cfg)], [("best", None, None, 1.0)]
    for n in free:
        for f in FACTORS:
            jobs.append(({**base_d, n: base_d[n] * float(f)}, cfg)); tags.append(("profile", None, n, float(f)))
    for g, sel in GROUPS.items():
        names = [n for n in free if sel(n)]
        if not names:
            continue
        for s in SIGMAS:
            for _ in range(args.n_joint):
                d = dict(base_d)
                for n in names:
                    d[n] = base_d[n] * float(np.exp(rng.normal(0, s)))
                jobs.append((d, cfg)); tags.append(("joint", g, None, s))

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_eval, jobs, chunksize=8))
    print(f"{len(jobs)} evals in {time.time() - t0:.0f}s")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [dict(kind=k, group=g, param=n, factor=f, loss=r[0], bistable=r[1], **r[2])
            for (k, g, n, f), r in zip(tags, res)]
    (out / "robustness.json").write_text(json.dumps(dict(
        fit_dir=str(fit_dir), free=free,
        bounds={n: [bounds[n].lo, bounds[n].hi] for n in free},
        best={n: base_d[n] for n in free},
        targets={"low_pyr": cfg.r_low_target, "low_som": cfg.r_som_target,
                 "low_pv": cfg.r_pv_target, "low_vip": cfg.r_vip_target,
                 "high_pyr": cfg.r_pyr_high_target, "high_som": cfg.r_som_high_target,
                 "high_pv": cfg.r_pv_high_target, "high_vip": cfg.r_vip_high_target},
        rows=rows), indent=1))
    print("wrote", out / "robustness.json")


if __name__ == "__main__":
    main()
