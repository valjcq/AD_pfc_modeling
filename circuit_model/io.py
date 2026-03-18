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
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .params import CircuitParams
    from .loss import TargetRates


def output_dir(base_dir: str, params_json: str) -> str:
    """Derive output directory from params file: base_dir/<stem>/ or base_dir/default/."""
    if params_json:
        stem = Path(params_json).stem
    else:
        stem = "default"
    out = os.path.join(base_dir, stem)
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


def save_params_json(path: str, params: "CircuitParams") -> None:
    """Save CircuitParams to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2, sort_keys=True)


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
) -> None:
    """
    Log optimization result to a JSONL file.

    Each line is a JSON object with step, loss, target, means, ko_means, and params.
    """
    entry = {
        "step": step,
        "loss": loss,
        "target": asdict(target),
        "means": means,
        "ko_means": ko_means,
        "params": asdict(params),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
