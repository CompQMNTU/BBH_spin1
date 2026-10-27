"""Theta scan by imaginary-time evolution, dense (SPEC §6 M3); sqrt3 x sqrt3 cell by default.

With NNN couplings (J2, K2; M5) the cell must be a patch such as (3, 3) or (6, 6): NNN gates cannot
be routed on sqrt3 (lattice.gate_paths).

For every theta and initial state (FM, FQ, AFQ, SU3 seeds, random product) evolve at bond
dimension D and append one JSON line per run to the output file. Runs already in the file are
skipped, so a scan can be resumed. Warm start: ``only_best`` keeps, per theta, only the
lowest-energy start of an earlier scan (:func:`best_per_theta`); ``save_dir`` pickles the final
states; ``from_dir`` starts from such saved states instead of the product states.
"""

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import io

INITS = ("FM", "FQ", "AFQ", "SU3", "random")
TIE = 1e-8
DEFAULT_SCHEDULE = [0.1, 0.05, 0.02, 0.01]  # evolve.DEFAULT_SCHEDULE; records without 'schedule' used it


def run_params(r):
    """Calculation parameters of a scan record; records written before 'schedule' and
    'warm_start' were stored used the default schedule and no warm start."""
    return {"n_theta": r["n_theta"], "D": r["D"], "env": r["env"],
            "schedule": r.get("schedule", DEFAULT_SCHEDULE), "warm_start": r.get("warm_start"),
            "J2": r.get("J2", 0.0), "K2": r.get("K2", 0.0), "cell": r.get("cell", "sqrt3"), "cell_dims": r.get("cell_dims")}


def best_per_theta(path):
    """Lowest-energy run per theta of a scan file. Runs within TIE of the minimum count as
    degenerate (e.g. the FM multiplet, where a random start reaches the FM energy with a reduced
    moment); among those the earliest entry of INITS wins, so seeds beat the random start."""
    runs = defaultdict(list)
    with open(path) as f:
        for r in map(json.loads, f):
            runs[r["theta_k"]].append(r)
    out = []
    for k in sorted(runs):
        e_min = min(r["energy"] for r in runs[k])
        cands = [r for r in runs[k] if r["energy"] <= e_min + TIE]
        out.append(min(cands, key=lambda r: INITS.index(r["init"])))
    return out


def state_file(directory, theta_k, init):
    return os.path.join(directory, f"k{theta_k:03d}_{init}.pkl")


def run_one(theta_k, n_theta, D, init, env, schedule, from_dir, save_dir, J2=0.0, K2=0.0, cell="sqrt3", cell_dims=None):
    from . import evolve, lattice, measure, operators, states
    from .hamiltonian import Model

    theta = 2 * np.pi * theta_k / n_theta
    ops = operators.spin1("dense")
    geo = lattice.geometry(cell, cell_dims)
    if from_dir:
        psi = io.load_state(state_file(from_dir, theta_k, init), ops.config)
    elif init == "random":
        psi = states.random_product_peps(geo, ops, seed=theta_k)
    else:
        psi = states.product_peps(geo, states.seed_vectors(init), ops)
    r = evolve.evolve(psi, Model(theta, J2=J2, K2=K2), ops, D, env=env, schedule=schedule)
    order = measure.order_parameters(psi, measure.ctm_env(psi, evolve.default_chi(D)), ops)
    if save_dir:
        io.save_state(state_file(save_dir, theta_k, init), psi)
    return {
        "theta_k": theta_k, "n_theta": n_theta, "theta_over_pi": 2 * theta_k / n_theta, "D": D,
        "env": env, "init": init, "warm_start": from_dir, "schedule": list(schedule),
        "J2": J2, "K2": K2, "cell": cell, "cell_dims": list(cell_dims) if cell_dims else None,
        "energy": r.energy, "class_energies": r.class_energies,
        "m": [float(m) for m, _ in order.values()], "Q": [float(q) for _, q in order.values()],
        "stages": [vars(s) for s in r.stages], "wall_time": r.wall_time,
    }


def run_scan(D, out, n_theta=80, env="BP", workers=8, schedule=None, only_best=None, from_dir=None, save_dir=None,
             log=print, J2=0.0, K2=0.0, cell="sqrt3", cell_dims=None):
    """Run (or resume) a scan; returns {"done": runs written now, "failed": [(theta_k, init, error)]}.
    Each record gets ``metadata`` (io.metadata of the scan process)."""
    from .evolve import DEFAULT_SCHEDULE as default
    schedule = tuple(schedule) if schedule else default
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    params = {"n_theta": n_theta, "D": D, "env": env, "schedule": list(schedule), "warm_start": from_dir,
              "J2": J2, "K2": K2, "cell": cell, "cell_dims": list(cell_dims) if cell_dims else None}
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for r in map(json.loads, f):
                bad = {k: v for k, v in run_params(r).items() if v != params[k]}
                if bad:
                    raise ValueError(f"{out} holds runs with {bad}, not {params}; use another output file")
                done.add((r["theta_k"], r["init"]))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if only_best:
        best = best_per_theta(only_best)
        if any(r["n_theta"] != n_theta for r in best):
            raise ValueError(f"{only_best} uses a different theta grid; pass its n_theta")
        jobs = [(r["theta_k"], r["init"]) for r in best]
    else:
        jobs = [(k, i) for k in range(n_theta) for i in INITS]
    jobs = [j for j in jobs if j not in done]
    meta = io.metadata()
    written, failed = 0, []
    with ProcessPoolExecutor(workers) as pool, open(out, "a") as f:
        futs = {pool.submit(run_one, k, n_theta, D, i, env, schedule, from_dir, save_dir, J2, K2, cell, cell_dims): (k, i)
                for k, i in jobs}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps({**fut.result(), "metadata": meta}) + "\n")
                f.flush()
                written += 1
            except Exception as e:  # keep the scan going; report the failed run
                failed.append((*futs[fut], repr(e)))
                log(f"failed {futs[fut]} {e!r}")
    return {"done": written, "failed": failed}
