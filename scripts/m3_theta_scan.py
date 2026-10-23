"""M3 (c): theta scan of the NN model, sqrt3 x sqrt3 cell, dense, EnvBP evolution (SPEC §6).

Thin wrapper around ``bbh_tri.scan.run_scan`` (also ``bbh-run`` with ``task: scan``). For every
theta and initial state (random product, FM, FQ, AFQ, SU3 seeds) evolve at bond dimension D and
append one JSON line per run to the output file. Runs already in the file are skipped, so the
scan can be resumed.

    uv run python scripts/m3_theta_scan.py --D 4 --out results/m3_scan_D4.jsonl --workers 16

Warm start (e.g. NTU refinement of the BP phase diagram): ``--only-best SCAN`` keeps, per theta,
only the lowest-energy start of an earlier scan (``best_per_theta``); ``--save-dir`` pickles the
final states; ``--from-dir`` starts from such saved states instead of the product states.

    uv run python scripts/m3_theta_scan.py --D 4 --only-best results/m3_scan_D4.jsonl \
        --save-dir states/D4_BP --out results/m3_best_D4_BP.jsonl --workers 16
    uv run python scripts/m3_theta_scan.py --D 4 --env NTU --only-best results/m3_scan_D4.jsonl \
        --from-dir states/D4_BP --schedule 0.02 0.01 --out results/m3_scan_D4_NTU.jsonl --workers 16
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse


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
    from bbh_tri.scan import run_scan
    try:
        run_scan(a.D, a.out, n_theta=a.n_theta, env=a.env, workers=a.workers, schedule=a.schedule,
                 only_best=a.only_best, from_dir=a.from_dir, save_dir=a.save_dir)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
