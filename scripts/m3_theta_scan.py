"""M3 (c): theta scan of the NN model, sqrt3 x sqrt3 cell, dense, EnvBP evolution (SPEC §6).

For every theta and initial state (random product, FM, FQ, AFQ, SU3 seeds) evolve at bond
dimension D and append one JSON line per run to the output file. Runs already in the file are
skipped, so the scan can be resumed.

    uv run python scripts/m3_theta_scan.py --D 4 --out results/m3_scan_D4.jsonl --workers 40

Warm start (e.g. NTU refinement of the BP phase diagram): ``--only-best SCAN`` keeps, per theta,
only the lowest-energy start of an earlier scan (``best_per_theta``); ``--save-dir`` pickles the
final states; ``--from-dir`` starts from such saved states instead of the product states.

    uv run python scripts/m3_theta_scan.py --D 4 --only-best results/m3_scan_D4.jsonl \
        --save-dir states/D4_BP --out results/m3_best_D4_BP.jsonl --workers 40
    uv run python scripts/m3_theta_scan.py --D 4 --env NTU --only-best results/m3_scan_D4.jsonl \
        --from-dir states/D4_BP --schedule 0.02 0.01 --out results/m3_scan_D4_NTU.jsonl --workers 40
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

INITS = ("FM", "FQ", "AFQ", "SU3", "random")
TIE = 1e-8
DEFAULT_SCHEDULE = [0.1, 0.05, 0.02, 0.01]  # bbh_tri.evolve.DEFAULT_SCHEDULE


def run_params(r):
    """Calculation parameters of a scan record; records written before 'schedule' and
    'warm_start' were stored used the default schedule and no warm start."""
    return {"n_theta": r["n_theta"], "D": r["D"], "env": r["env"],
            "schedule": r.get("schedule", DEFAULT_SCHEDULE), "warm_start": r.get("warm_start")}


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


def run_one(theta_k, n_theta, D, init, env, schedule, from_dir, save_dir):
    import yastn.tn.fpeps as fpeps

    from bbh_tri import evolve, lattice, measure, operators, states
    from bbh_tri.hamiltonian import Model
    from bbh_tri.yastn_compat import peps_to_dict

    theta = 2 * np.pi * theta_k / n_theta
    ops = operators.spin1("dense")
    geo = lattice.geometry("sqrt3")
    if from_dir:
        with open(state_file(from_dir, theta_k, init), "rb") as f:
            psi = fpeps.Peps.from_dict(pickle.load(f), config=ops.config)
    elif init == "random":
        psi = states.random_product_peps(geo, ops, seed=theta_k)
    else:
        psi = states.product_peps(geo, states.seed_vectors(init), ops)
    r = evolve.evolve(psi, Model(theta), ops, D, env=env, schedule=schedule)
    order = measure.order_parameters(psi, measure.ctm_env(psi, evolve.default_chi(D)), ops)
    if save_dir:
        with open(state_file(save_dir, theta_k, init), "wb") as f:
            pickle.dump(peps_to_dict(psi), f)
    return {
        "theta_k": theta_k, "n_theta": n_theta, "theta_over_pi": 2 * theta_k / n_theta, "D": D,
        "env": env, "init": init, "warm_start": from_dir, "schedule": list(schedule),
        "energy": r.energy, "class_energies": r.class_energies,
        "m": [float(m) for m, _ in order.values()], "Q": [float(q) for _, q in order.values()],
        "stages": [vars(s) for s in r.stages], "wall_time": r.wall_time,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, required=True)
    p.add_argument("--n-theta", type=int, default=80)
    p.add_argument("--env", default="BP")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--schedule", type=float, nargs="+", default=None)
    p.add_argument("--only-best", default=None, help="scan file; run only its best start per theta")
    p.add_argument("--from-dir", default=None, help="start from states saved by --save-dir")
    p.add_argument("--save-dir", default=None)
    a = p.parse_args()
    from bbh_tri.evolve import DEFAULT_SCHEDULE
    schedule = tuple(a.schedule) if a.schedule else DEFAULT_SCHEDULE
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)

    params = {"n_theta": a.n_theta, "D": a.D, "env": a.env, "schedule": list(schedule), "warm_start": a.from_dir}
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for r in map(json.loads, f):
                bad = {k: v for k, v in run_params(r).items() if v != params[k]}
                if bad:
                    raise SystemExit(f"{a.out} holds runs with {bad}, not {params}; use another --out")
                done.add((r["theta_k"], r["init"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.only_best:
        best = best_per_theta(a.only_best)
        if any(r["n_theta"] != a.n_theta for r in best):
            raise SystemExit(f"{a.only_best} uses a different theta grid; pass its --n-theta")
        jobs = [(r["theta_k"], r["init"]) for r in best]
    else:
        jobs = [(k, i) for k in range(a.n_theta) for i in INITS]
    jobs = [j for j in jobs if j not in done]
    with ProcessPoolExecutor(a.workers) as pool, open(a.out, "a") as f:
        futs = {pool.submit(run_one, k, a.n_theta, a.D, i, a.env, schedule, a.from_dir, a.save_dir): (k, i) for k, i in jobs}
        for fut in as_completed(futs):
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except Exception as e:  # keep the scan going; report the failed run
                print("failed", futs[fut], repr(e), flush=True)


if __name__ == "__main__":
    main()
