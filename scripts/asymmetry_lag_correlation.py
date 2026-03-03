"""
Lagged pre-cue correlation analysis  (options 2 & 3).

For each condition, re-runs only the burn-in phase of each trial (up to cue
onset) to obtain the full A(t) timecourse, then:

  Option 2 — Lagged correlation curve
    Computes  ρ(lag) = Pearson r( A(t_cue - lag),  delay_asym )
    as a function of lag (0 … MAX_LAG_MS ms before the cue).
    If the instantaneous pre-cue state matters, ρ should peak near lag = 0
    and decay as lag grows.

  Option 3 — Conditional distributions
    Splits trials by the sign of A at the last time step before the cue
    (rightward vs leftward), and compares the delay_asym distributions of
    each group per condition.

Usage:
    python scripts/asymmetry_lag_correlation.py [--n_trials N] [--max_lag MS]

The script reads delay_asym from the existing CSV (no need to re-run the
full cue+delay simulation), and only re-runs the burn-in phase for the
pre-cue timecourse.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu, pearsonr
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Configuration — must match the experiment CSV
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(
    ROOT,
    "figs/asymmetry/128/default/gauss_w7_s30-pv_unif_10/amp45_uncorrected",
    "asymmetry_trials.csv",
)
OUT_DIR  = os.path.dirname(CSV_PATH)

MAX_LAG_MS   = 1000.0   # how far before the cue to probe (ms)
RECORD_DT_MS = 5.0      # temporal resolution of the lag axis

# Simulation parameters (must match the experiment)
SETTLING_MS      = 6000.0   # burn-in duration = cue onset time
STIM_DURATION_MS = 250.0
STIM_SIGMA_DEG   = 18.0
AMP_FACTOR       = 45.0
DELAY_MS         = 5000.0
N_NODES          = 128
W_PYR_INTER      = 7.0
SIGMA_PYR_DEG    = 30.0
W_PV_GLOBAL      = 10.0

CONDITIONS = ["WT", "WT_APP", "a7_KO_APP"]
COLORS     = {"WT": "#2196F3", "WT_APP": "#FF9800", "a7_KO_APP": "#F44336"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=None,
                    help="Max trials per condition (default: all)")
parser.add_argument("--max_lag",  type=float, default=MAX_LAG_MS,
                    help=f"Lag range in ms (default: {MAX_LAG_MS})")
parser.add_argument("--workers",  type=int, default=None,
                    help="Parallel workers (default: cpu_count)")
args = parser.parse_args()

MAX_LAG_MS = args.max_lag

# ---------------------------------------------------------------------------
# 1. Load CSV — collect seeds and delay metrics per condition
# ---------------------------------------------------------------------------
trials_by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        cond = row["condition"]
        if cond not in CONDITIONS:
            continue
        trials_by_cond[cond].append({
            "seed":       int(row["seed"]),
            "trial_idx":  int(row["trial_idx"]),
            "delay_asym": float(row["delay_asym"]),
        })

for cond in CONDITIONS:
    trials_by_cond[cond].sort(key=lambda r: r["trial_idx"])
    if args.n_trials is not None:
        trials_by_cond[cond] = trials_by_cond[cond][: args.n_trials]
    n = len(trials_by_cond[cond])
    print(f"  {cond}: {n} trials")

# ---------------------------------------------------------------------------
# 2. Set up the circuit (shared across all workers via module-level state)
# ---------------------------------------------------------------------------
from circuit_model.params import CircuitParams
from circuit_model.study import STUDY_CONDITIONS, apply_condition
from circuit_model.ring.params import RingParams
from circuit_model.ring.connectivity import RingConnectivity
from circuit_model.ring.simulation import simulate_ring
from circuit_model.ring.analysis import compute_bump_asymmetry

base_params  = CircuitParams()
ring_params  = RingParams(
    n_nodes=N_NODES,
    w_pyr_pyr_inter=W_PYR_INTER,
    sigma_pyr_deg=SIGMA_PYR_DEG,
    w_pv_global=W_PV_GLOBAL,
)
connectivity = RingConnectivity.from_params(ring_params)

# Pre-compute per-condition local_params (no rng → mean values, same as CLI)
local_params_by_cond = {
    cond: apply_condition(base_params, STUDY_CONDITIONS[cond])
    for cond in CONDITIONS
}

# ---------------------------------------------------------------------------
# Worker — run only the burn-in phase, return last MAX_LAG_MS of A(t)
# ---------------------------------------------------------------------------
_worker_state = {}

def _init_worker(lp_by_cond, rp, conn, record_dt, settling, max_lag):
    global _worker_state
    _worker_state = {
        "lp_by_cond": lp_by_cond,
        "ring_params": rp,
        "connectivity": conn,
        "record_dt_ms": record_dt,
        "settling_ms": settling,
        "max_lag_ms": max_lag,
    }

def _run_precue(job: tuple) -> tuple:
    """Run burn-in only, return A(t) for the last max_lag_ms before cue."""
    cond_key, seed = job
    ws = _worker_state

    result = simulate_ring(
        ws["lp_by_cond"][cond_key],
        ws["ring_params"],
        T_ms=ws["settling_ms"],
        stimuli=None,
        seed=seed,
        connectivity=ws["connectivity"],
        record_dt_ms=ws["record_dt_ms"],
    )
    asym  = compute_bump_asymmetry(result)   # (n_steps,)
    t_ms  = result.t_ms
    del result

    # Keep only the last max_lag_ms window
    t_cut = t_ms[-1] - ws["max_lag_ms"]
    mask  = t_ms >= t_cut
    return seed, asym[mask]   # (n_lag_steps,)


# ---------------------------------------------------------------------------
# 3. Run simulations in parallel per condition
# ---------------------------------------------------------------------------
n_lag_steps = int(round(MAX_LAG_MS / RECORD_DT_MS)) + 1
# lag axis: 0 ms = last step before cue, MAX_LAG_MS = furthest back
lag_ms = np.linspace(0, MAX_LAG_MS, n_lag_steps)

precue_by_cond: dict[str, np.ndarray] = {}  # (n_trials, n_lag_steps)
n_workers = args.workers

for cond in CONDITIONS:
    trials = trials_by_cond[cond]
    jobs   = [(cond, t["seed"]) for t in trials]
    traj_dict: dict[int, np.ndarray] = {}

    print(f"\nRunning burn-in for {cond} ({len(jobs)} trials)…")
    init_args = (local_params_by_cond, ring_params, connectivity,
                 RECORD_DT_MS, SETTLING_MS, MAX_LAG_MS)

    if n_workers != 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker,
                                 initargs=init_args) as ex:
            futures = {ex.submit(_run_precue, j): j for j in jobs}
            with tqdm(total=len(jobs), unit="trial") as pbar:
                for fut in as_completed(futures):
                    seed, traj = fut.result()
                    traj_dict[seed] = traj
                    pbar.update()
    else:
        _init_worker(*init_args)
        for job in tqdm(jobs, unit="trial"):
            seed, traj = _run_precue(job)
            traj_dict[seed] = traj

    # Stack into (n_trials, n_lag_steps), aligning from the end
    rows = []
    for t in trials:
        traj = traj_dict[t["seed"]]
        # traj[-1] = last point before cue (lag 0); traj[-n] = lag (n-1)*dt
        if len(traj) >= n_lag_steps:
            rows.append(traj[-n_lag_steps:][::-1])   # index 0 = lag 0
        else:
            pad = np.full(n_lag_steps, np.nan)
            pad[:len(traj)] = traj[::-1]
            rows.append(pad)
    precue_by_cond[cond] = np.array(rows)   # (n_trials, n_lag_steps)

# ---------------------------------------------------------------------------
# 4. Compute lagged Pearson r and 95 % bootstrap CI
# ---------------------------------------------------------------------------
def _bootstrap_r(x, y, n_boot=500, rng=None):
    """Return (r, ci_lo, ci_hi) with 95 % percentile bootstrap CI."""
    rng = rng or np.random.default_rng(0)
    n = len(x)
    r_obs, _ = pearsonr(x, y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots[i], _ = pearsonr(x[idx], y[idx])
        except Exception:
            boots[i] = np.nan
    return r_obs, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))

lag_corr: dict[str, dict] = {}
for cond in CONDITIONS:
    mat   = precue_by_cond[cond]           # (n_trials, n_lag_steps)
    y_all = np.array([t["delay_asym"] for t in trials_by_cond[cond]])
    rs, lo, hi = [], [], []
    rng = np.random.default_rng(42)
    for lag_i in range(n_lag_steps):
        x = mat[:, lag_i]
        valid = ~(np.isnan(x) | np.isnan(y_all))
        if valid.sum() > 5:
            r, ci_lo, ci_hi = _bootstrap_r(x[valid], y_all[valid], rng=rng)
        else:
            r, ci_lo, ci_hi = np.nan, np.nan, np.nan
        rs.append(r); lo.append(ci_lo); hi.append(ci_hi)
    lag_corr[cond] = {
        "r":  np.array(rs),
        "lo": np.array(lo),
        "hi": np.array(hi),
    }

# ---------------------------------------------------------------------------
# 5. Conditional distributions: sign of last A vs delay_asym
# ---------------------------------------------------------------------------
# last_pre_cue = mat[:, 0] (lag = 0)
cond_dist: dict[str, dict] = {}
for cond in CONDITIONS:
    last_a = precue_by_cond[cond][:, 0]
    y_all  = np.array([t["delay_asym"] for t in trials_by_cond[cond]])
    right  = y_all[last_a > 0]
    left   = y_all[last_a < 0]
    p_mwu  = np.nan
    if len(right) > 1 and len(left) > 1:
        _, p_mwu = mannwhitneyu(right, left, alternative="two-sided")
    cond_dist[cond] = {"right": right, "left": left, "p_mwu": p_mwu}

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs  = GridSpec(2, len(CONDITIONS), figure=fig, hspace=0.45, wspace=0.35)

# --- Top row: single wide panel for ρ(lag), spanning all columns ---
ax_lag = fig.add_subplot(gs[0, :])
for cond in CONDITIONS:
    lc = lag_corr[cond]
    c  = COLORS[cond]
    label = STUDY_CONDITIONS[cond].name
    ax_lag.plot(lag_ms, lc["r"], color=c, lw=2.0, label=label)
    ax_lag.fill_between(lag_ms, lc["lo"], lc["hi"], color=c, alpha=0.15)

ax_lag.axhline(0, color="k", lw=0.8, ls="--")
ax_lag.axvline(0, color="k", lw=0.5, ls=":", alpha=0.5)
ax_lag.set_xlabel("Lag before cue onset (ms)", fontsize=11)
ax_lag.set_ylabel("Pearson r  [A(t_cue − lag) vs delay asym]", fontsize=10)
ax_lag.set_title(
    "Option 2 — Lagged correlation\n"
    "Does the pre-cue state at lag τ predict the delay asymmetry?",
    fontsize=10, fontweight="bold",
)
ax_lag.legend(fontsize=9, loc="upper right")
ax_lag.set_xlim(0, MAX_LAG_MS)
ax_lag.spines["top"].set_visible(False)
ax_lag.spines["right"].set_visible(False)

# --- Bottom row: conditional distributions per condition ---
def _stars(p):
    if np.isnan(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

for col_i, cond in enumerate(CONDITIONS):
    ax = fig.add_subplot(gs[1, col_i])
    cd = cond_dist[cond]
    c  = COLORS[cond]

    data   = [cd["right"], cd["left"]]
    labels = [f"Pre-cue\nrightward\n(n={len(cd['right'])})",
              f"Pre-cue\nleftward\n(n={len(cd['left'])})"]

    vp = ax.violinplot(data, positions=[0, 1], showmedians=True, widths=0.6)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(c)
        body.set_alpha(0.55)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color(c)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.7, ls="--")

    # Annotate with MWU p-value
    y_top = max(
        (np.nanmax(cd["right"]) if len(cd["right"]) else 0),
        (np.nanmax(cd["left"])  if len(cd["left"])  else 0),
    ) * 1.1
    ax.text(0.5, y_top, _stars(cd["p_mwu"]),
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            transform=ax.get_xaxis_transform())

    if col_i == 0:
        ax.set_ylabel("Delay asymmetry", fontsize=10)
    ax.set_title(
        f"{STUDY_CONDITIONS[cond].name}\n"
        f"MWU p = {cd['p_mwu']:.3f}" if not np.isnan(cd["p_mwu"]) else
        f"{STUDY_CONDITIONS[cond].name}",
        fontsize=9, fontweight="bold", color=c,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Pre-cue state → delay asymmetry: lagged correlation & conditional distributions\n"
    f"({len(trials_by_cond[CONDITIONS[0]])} trials/condition, "
    f"n_nodes={N_NODES}, w_inter={W_PYR_INTER}, σ={SIGMA_PYR_DEG}°, "
    f"amp={AMP_FACTOR}×, uncorrected)",
    fontsize=11,
)

out_path = os.path.join(OUT_DIR, "pre_cue_lag_correlation.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
plt.show()
