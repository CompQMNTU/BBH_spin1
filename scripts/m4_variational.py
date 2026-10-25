"""M4: variational (AD) optimization of one chain, then validation (docs/ad_workflow.md).

Thin wrapper around ``bbh_tri.runs.run_variational`` (also ``bbh-run`` with ``task: variational``).

    uv run --extra ad python scripts/m4_variational.py --theta-over-pi 0.25 --D 3 \
        --seed states/pi4_D3_BP.pkl --out results/m4/pi4_D3

Seeds (``bbh_tri.runs`` docstring): a saved state (.pkl, or a chain checkpoint .pt), padded to D
with ``--noise`` if dense and smaller; ``bp`` (EnvBP at D from a random product state); ``bp:NAME``
(EnvBP at D from the product seed NAME: FM, FQ, AFQ, SU3); ``bp:PATH`` (EnvBP at D from a saved
state, e.g. truncating a D+1 state to D); or a product seed name (padded to D).

``--sym`` (SPEC §4.5): dense, U1, U1xU1 (theta = pi/4 only) or Z2xZ2. Symmetric tensors cannot be
padded, so a symmetric run needs ``bp:NAME`` with a compatible seed (U1xU1: bp:SU3, Z2xZ2:
bp:AFQ, U1: bp:FQ), ``bp:PATH``, or a state saved with the same symmetry at bond dimension D.
Writes OUT.pt (checkpoint; resumed if present, an error if it was made for another theta, chi or
D), OUT.pkl (final numpy state) and OUT.json (history, validation at chi = 2D^2, 3D^2, 4D^2,
metadata: git hash, versions, host).

``--chi`` defaults to 3 D^2. At 2 D^2 the cut can fall inside a near-degenerate group of
environment singular values; the gap-safe truncation keeps or drops the group whole, so the loss
jumps by ~1e-7 (U1xU1, D = 5) where the grouping changes along a search direction. That is as large
as the per-step gains near convergence, and the line search stalls. At 3 D^2, D = 5 converges;
D = 4 still stalls with <~1e-6 gain per step left, and moves again at 4 D^2 (docs/ad_workflow.md §2).
Several values (``--chi 3D2 4D2``) are stages: when a stage stalls (``Chain.run``: many line-search
evaluations for almost no gain), the chain continues at the next chi from its last state, with
checkpoint OUT.chiN.pt. A checkpoint resumes only at its own chi.

``--threads N`` (default 1) sets the BLAS/OpenMP and torch intra-op threads. It pays only for dense
configs (D = 4: 2.3x at 4 threads, nothing more at 8); U1xU1 spends its time in per-block Python
loops and gains nothing. Keep the node's total within its worker budget.
``--max-sweeps`` caps the CTM sweeps per evaluation. Unrolled AD keeps every sweep for the
backward pass, so memory grows with the sweep count; an evaluation that does not converge within
the cap fails the step, which is retried after a perturbation. The log records the sweeps of each
evaluation. A checkpoint may be resumed with another --max-sweeps.
``--ad fp`` differentiates the CTM fixed point implicitly (``fp_ctmrg``) instead of backpropagating
through every sweep: memory no longer grows with the sweeps (theta = 0, dense D = 5, chi = 75:
8.4 GB on a GPU, against ~300 GB unrolled on CPU). ``--max-sweeps`` then caps the forward CTM sweeps
and the terms of the backward fixed-point series. A checkpoint resumes only with its own --ad; to
switch, start another --out with --seed OUT.pt (docs/ad_workflow.md §3).
``--max-hours`` stops cleanly before a batch-system time limit (status 'time', no validation);
rerun the same command to resume.
"""

import argparse
import os


def _threads(argv):
    """--threads before numpy/torch load (BLAS reads these variables once, at import)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--threads", type=int, default=1)
    return p.parse_known_args(argv)[0].threads


THREADS = _threads(None)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--theta-over-pi", type=float, required=True)
    p.add_argument("--D", type=int, required=True)
    p.add_argument("--seed", required=True, help="PATH | bp | bp:NAME | bp:PATH | FM | FQ | AFQ | SU3 | random")
    p.add_argument("--sym", default="dense", help="dense | U1 | U1xU1 | Z2xZ2")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--chi", nargs="+", default=None, help="optimization chi stage(s), e.g. 75 or 3D2 4D2 (default 3D2)")
    p.add_argument("--max-steps", type=int, default=200, help="steps per chi stage")
    p.add_argument("--noise", type=float, default=1e-3)
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP/torch threads (pays for dense only)")
    p.add_argument("--max-sweeps", type=int, default=100, help="CTM sweeps per evaluation (bounds AD memory)")
    p.add_argument("--ad", default="unrolled", help="unrolled | fp (gradient through the CTM)")
    p.add_argument("--device", default="cpu", help="torch device for the AD chain, e.g. cuda:0")
    p.add_argument("--max-hours", type=float, default=None, help="stop cleanly after this wall time")
    p.add_argument("--stall-window", type=int, default=5, help="steps in the stall test (0 disables it)")
    p.add_argument("--no-validate", action="store_true")
    a = p.parse_args()

    from bbh_tri.runs import run_variational
    chi = None if a.chi is None else [c if c.upper().endswith("D2") else int(c) for c in a.chi]
    cfg = {"theta_over_pi": a.theta_over_pi, "D": a.D, "seed": a.seed, "sym": a.sym, "out": a.out, "chi": chi,
           "max_steps": a.max_steps, "noise": a.noise, "rng": a.rng, "threads": a.threads, "max_sweeps": a.max_sweeps,
           "ad": a.ad, "device": a.device, "max_hours": a.max_hours, "stall_window": a.stall_window, "validate": not a.no_validate}
    try:
        rec = run_variational(cfg, log=lambda s: print(s, flush=True))
    except ValueError as e:
        raise SystemExit(str(e))
    print(rec["status"], json.dumps(rec.get("validation", {}).get("energies")), flush=True)


if __name__ == "__main__":
    main()
