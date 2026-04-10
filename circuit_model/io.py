"""
Input/output utilities for the circuit model.

This module contains:
- load_params_json: Load parameters from JSON file
- save_params_json: Save parameters to JSON file
- log_best_result: Log optimization results to JSONL file
- format_params_as_code: Format parameters as Python code
"""

from __future__ import annotations

import os
from dataclasses import asdict, fields
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json

import numpy as np

if TYPE_CHECKING:
    from .params import CircuitParams
    from .loss import TargetRates
    from .optimization import KOMeans


def output_dir(base_dir: str, params_json: str) -> str:
    """Return a flat output directory path (no params-based subfolders)."""
    _ = params_json  # Kept for backward-compatible call sites.
    out = base_dir
    os.makedirs(out, exist_ok=True)
    return out


def load_params_json(path: str) -> "CircuitParams":
    """Load CircuitParams from a JSON file."""
    from .params import CircuitParams

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # Handle nested {"params": {...}} format (e.g. from log_best_result)
    if "params" in d and isinstance(d["params"], dict):
        d = d["params"]
    base = CircuitParams()
    allowed = {fld.name for fld in fields(CircuitParams)}
    clean = {k: d[k] for k in d if k in allowed}
    return replace(base, **clean)


def load_ring_params_json(path: str) -> "RingParams":
    """Load RingParams from a JSON file."""
    from .ring.params import RingParams

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    allowed = {fld.name for fld in fields(RingParams)}
    clean = {k: d[k] for k in d if k in allowed}
    return RingParams(**clean)


def build_fit_comparison(
    means: np.ndarray,
    ko_means: "KOMeans",
    target: "TargetRates",
    loss: float,
    jacobian: Optional[np.ndarray] = None,
    display_ko_targets: Optional["TargetRates"] = None,
) -> dict:
    """
    Build fit metadata dict: actual vs target comparison for each condition/population,
    plus optional Jacobian effective-gain matrix.

    Returns a dict with keys:
    - ``loss``: total optimization loss
    - ``comparison``: nested dict {condition: {pop: {actual, target, error_pct}}}
    - ``jacobian`` (if provided): {pop: {pop: gain}} where J[i,j] = dr_i/dr_j
    """
    pops = ["PYR", "SOM", "PV", "VIP"]
    tgt_arr = target.as_array()

    def _entry(actual: float, tgt: float) -> dict:
        err_pct = round(100.0 * (actual - tgt) / max(abs(tgt), 1e-6), 2)
        return {"actual": round(actual, 4), "target": round(tgt, 4), "error_pct": err_pct}

    def _entry_info(actual: float, tgt: Optional[float] = None) -> dict:
        """KO entry shown as info only (not in loss). Target shown if provided."""
        err_pct = round(100.0 * (actual - tgt) / max(abs(tgt), 1e-6), 2) if tgt is not None else None
        return {"actual": round(actual, 4), "target": round(tgt, 4) if tgt is not None else None, "error_pct": err_pct, "in_loss": False}

    comparison: dict = {
        "base": {pop: _entry(float(means[i]), float(tgt_arr[i])) for i, pop in enumerate(pops)},
    }
    if ko_means.alpha7_ko is not None:
        if target.alpha7_ko_pyr is not None:
            comparison["alpha7_ko"] = {"PYR": _entry(float(ko_means.alpha7_ko[0]), target.alpha7_ko_pyr)}
        else:
            comparison["alpha7_ko"] = {"PYR": _entry_info(float(ko_means.alpha7_ko[0]), display_ko_targets.alpha7_ko_pyr if display_ko_targets else None)}
    if ko_means.alpha5_ko is not None:
        if target.alpha5_ko_pyr is not None:
            comparison["alpha5_ko"] = {"PYR": _entry(float(ko_means.alpha5_ko[0]), target.alpha5_ko_pyr)}
        else:
            comparison["alpha5_ko"] = {"PYR": _entry_info(float(ko_means.alpha5_ko[0]), display_ko_targets.alpha5_ko_pyr if display_ko_targets else None)}
    if ko_means.beta2_ko is not None:
        if target.beta2_ko_pyr is not None:
            comparison["beta2_ko"] = {"PYR": _entry(float(ko_means.beta2_ko[0]), target.beta2_ko_pyr)}
        else:
            comparison["beta2_ko"] = {"PYR": _entry_info(float(ko_means.beta2_ko[0]), display_ko_targets.beta2_ko_pyr if display_ko_targets else None)}

    out: dict = {"loss": round(loss, 6), "comparison": comparison}

    if jacobian is not None:
        # J[i,j] = dr_i/dr_j  (row=target population, col=source population)
        out["jacobian"] = {
            "_note": "J[row,col] = dr_row/dr_col: effect of col population on row population",
            **{
                pops[i]: {pops[j]: round(float(jacobian[i, j]), 6) for j in range(4)}
                for i in range(4)
            },
        }

    return out


def save_params_json(
    path: str,
    params: "CircuitParams",
    fit_meta: Optional[dict] = None,
) -> None:
    """Save CircuitParams to a JSON file, with optional fit metadata.

    If ``fit_meta`` is provided (from :func:`build_fit_comparison`), it is stored
    under the ``_fit_metadata`` key at the top of the JSON so it is immediately
    visible. The key is ignored by :func:`load_params_json` (only CircuitParams
    fields are loaded).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sorted_params = dict(sorted(asdict(params).items()))
    out: dict = {}
    if fit_meta is not None:
        out["_fit_metadata"] = fit_meta
    out.update(sorted_params)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def save_fit_summary_txt(
    json_path: str,
    fit_meta: dict,
    params: Optional["CircuitParams"] = None,
) -> None:
    """Write a human-readable .txt summary alongside *json_path* (same stem, .txt extension).

    The file contains:
    - Loss value
    - Actual vs target firing rates for every fitted condition
    - Jacobian as a full 4×4 matrix
    - Jacobian as per-connection details with raw weight, effective gain, and strength label
      (requires *params* to access raw weight values)
    """
    import datetime

    txt_path = str(Path(json_path).with_suffix(".txt"))
    name = Path(json_path).name
    pops = ["PYR", "SOM", "PV", "VIP"]
    W = 80       # wide enough for connection-details lines
    SEP = "  " + "─" * (W - 2)

    lines: list[str] = []
    lines.append("=" * W)
    lines.append(f"  FIT SUMMARY  —  {name}")
    lines.append(f"  Generated: {datetime.date.today()}")
    lines.append("=" * W)

    loss = fit_meta.get("loss")
    if loss is not None:
        lines.append(f"  Loss: {loss:.4g}")
    lines.append("")

    # ── Steady-state rates (when no fit targets, just show simulated rates) ─
    ss = fit_meta.get("steady_state_rates")
    if ss:
        lines.append("  STEADY-STATE RATES  (simulated, used for Jacobian)")
        lines.append(SEP)
        lines.append("  " + "   ".join(f"{p}: {ss[p]:6.3f}" for p in pops if p in ss))
        lines.append(SEP)
        lines.append("")

    # ── Firing rate comparison (only when fit targets are available) ─────────
    cmp = fit_meta.get("comparison", {})
    if cmp:
        lines.append("  FIRING RATES: actual vs target")
        lines.append(SEP)
        lines.append(f"  {'Condition':<14}  {'Pop':<4}  {'Actual':>8}  {'Target':>8}  {'Error':>7}")
        lines.append(SEP)

        base = cmp.get("base", {})
        for pop in pops:
            if pop not in base:
                continue
            e = base[pop]
            lines.append(
                f"  {'base':<14}  {pop:<4}  {e['actual']:8.3f}  {e['target']:8.3f}  {e['error_pct']:+6.1f}%"
            )

        for cond_key in ("alpha7_ko", "alpha5_ko", "beta2_ko"):
            if cond_key not in cmp:
                continue
            lines.append(SEP)
            e = cmp[cond_key]["PYR"]
            if e["target"] is None:
                lines.append(
                    f"  {cond_key:<14}  {'PYR':<4}  {e['actual']:8.3f}  {'—':>8}  {'(info)':>7}"
                )
            elif e["error_pct"] is not None and not (e.get("in_loss", True)):
                lines.append(
                    f"  {cond_key:<14}  {'PYR':<4}  {e['actual']:8.3f}  {e['target']:8.3f}  {e['error_pct']:+6.1f}% (info)"
                )
            else:
                lines.append(
                    f"  {cond_key:<14}  {'PYR':<4}  {e['actual']:8.3f}  {e['target']:8.3f}  {e['error_pct']:+6.1f}%"
                )

        lines.append(SEP)
        lines.append("")

    # ── Transfer function parameters ────────────────────────────────────────
    if params is not None:
        lines.append("  TRANSFER FUNCTION  Phi(I) = c·(I−Θ) / (1 − exp(−g·c·(I−Θ)))")
        lines.append(f"  Curvature: g_exc (PYR) = {params.g_exc:.4f}   g_inh (SOM/PV/VIP) = {params.g_inh:.4f}")
        lines.append(SEP)
        lines.append(f"  {'':8}{'PYR':>10}{'SOM':>10}{'PV':>10}{'VIP':>10}")
        lines.append(SEP)
        lines.append(f"  {'Theta':8}{params.Theta_pyr:>10.3f}{params.Theta_som:>10.3f}{params.Theta_pv:>10.3f}{params.Theta_vip:>10.3f}")
        lines.append(f"  {'alpha':8}{params.alpha_pyr:>10.4f}{params.alpha_som:>10.4f}{params.alpha_pv:>10.4f}{params.alpha_vip:>10.4f}")
        lines.append(SEP)
        lines.append("")

    # ── Jacobian ────────────────────────────────────────────────────────────
    jac = fit_meta.get("jacobian")
    if jac:
        # Reconstruct J matrix from stored dict
        J = np.zeros((4, 4))
        for i, rp in enumerate(pops):
            for j, cp in enumerate(pops):
                J[i, j] = jac.get(rp, {}).get(cp, 0.0)

        # --- Matrix view ---
        lines.append("  JACOBIAN MATRIX  (J[row, col] = dr_row / dr_col)")
        lines.append("  Effect of source pop (col) on target pop (row) at fitted steady state")
        lines.append(SEP)
        col_w = 11
        lines.append("  " + " " * 6 + "".join(f"{p:>{col_w}}" for p in pops))
        lines.append(SEP)
        for i, row_pop in enumerate(pops):
            vals = "".join(f"{J[i, j]:+{col_w}.4f}" for j in range(4))
            lines.append(f"  {row_pop:<6}{vals}")
        lines.append(SEP)
        lines.append("")

        # --- Connection-details view ---
        lines.append("  CONNECTION DETAILS  (raw weight → effective gain)")
        lines.append(SEP)

        if params is not None:
            from .jacobian import _CONNECTIONS
            negligible_threshold = 0.005
            for (i, j, attr, desc, sign) in _CONNECTIONS:
                w_raw = getattr(params, attr)
                gain = J[i, j]
                abs_gain = abs(gain)
                if abs_gain > 0.1:
                    label = "STRONG    "
                elif abs_gain > 0.01:
                    label = "moderate  "
                elif abs_gain > negligible_threshold:
                    label = "weak      "
                else:
                    label = "NEGLIGIBLE ⚠"
                direction_ok = (gain > 0 and sign == "+") or (gain < 0 and sign == "-")
                dir_flag = "" if direction_ok else "  [WRONG SIGN ⚠]"
                lines.append(
                    f"  {desc:<44}  w={w_raw:8.3f}  J={gain:+.4f}  [{label}]{dir_flag}"
                )
        else:
            lines.append("  (params not available — load from JSON and regenerate for weight details)")

        lines.append(SEP)
        lines.append("")

    lines.append("=" * W)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_params_as_code(params: "CircuitParams") -> str:
    """Format CircuitParams as Python code for copy-paste."""
    from .params import CircuitParams

    lines = ["CircuitParams("]
    for f in fields(CircuitParams):
        name = f.name
        val = getattr(params, name)
        if isinstance(val, float):
            lines.append(f"    {name}={val:.6g},")
        else:
            lines.append(f"    {name}={val!r},")
    lines.append(")")
    return "\n".join(lines)


def log_best_result(
    path: str,
    step: int,
    loss: float,
    means: "dict",
    ko_means: "dict",
    params: "CircuitParams",
    target: "TargetRates",
    breakdown: Optional["dict"] = None,
) -> None:
    """
    Log optimization result to a JSONL file.

    Each line is a JSON object with step, loss, target, means, ko_means, params, and optionally breakdown.
    
    Args:
        breakdown: Optional dict with loss component breakdown {firing_rate, ko_firing_rate, jacobian, turing, ach_ratio, total}
    """
    entry = {
        "step": step,
        "loss": loss,
        "target": asdict(target),
        "means": means,
        "ko_means": ko_means,
        "params": asdict(params),
    }
    if breakdown is not None:
        entry["breakdown"] = breakdown
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
