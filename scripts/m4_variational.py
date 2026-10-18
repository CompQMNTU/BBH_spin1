"""M4: variational (AD) optimization of one chain, then validation (docs/ad_workflow.md).

    uv run --extra ad python scripts/m4_variational.py --theta-over-pi 0.25 --D 3 \
        --seed states/pi4_D3_BP.pkl --out results/m4/pi4_D3

Seeds: a saved numpy PEPS (``peps_to_dict`` pickle, e.g. from ``m3_theta_scan.py --save-dir``),
padded to D with ``--noise`` if its bond dimension is smaller; or ``bp`` (fresh EnvBP run at D
from a random product state) or a product seed name (FM, FQ, AFQ, SU3, random; padded to D).
Writes OUT.pt (checkpoint; resumed if present, an error if it was made for another theta, chi or D),
OUT.pkl (final numpy state) and OUT.json
(history, validation at chi = D^2, 2D^2, 3D^2).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import pickle

import numpy as np


def seed_state(spec, D, theta, noise, rng):
    import yastn.tn.fpeps as fpeps

    from bbh_tri import evolve, lattice, operators, states, variational as var
    from bbh_tri.hamiltonian import Model

    ops = operators.spin1("dense")
    geo = lattice.geometry("sqrt3")
    if os.path.exists(spec):
        with open(spec, "rb") as f:
            psi = fpeps.Peps.from_dict(pickle.load(f), config=ops.config)
    elif spec == "bp":
        psi = states.random_product_peps(geo, ops, seed=rng)
        evolve.evolve(psi, Model(theta), ops, D)
    elif spec == "random":
        psi = states.random_product_peps(geo, ops, seed=rng)
    else:
        psi = states.product_peps(geo, states.seed_vectors(spec), ops)
    dims = [d for s in psi.sites() for d in psi[s].get_shape()[:4]]
    if max(dims) > D:
        raise SystemExit(f"seed has bond dimension {max(dims)} > --D {D}")
    if min(dims) < D:
        psi = var.pad(psi, D, noise=noise, seed=rng)
    return psi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--theta-over-pi", type=float, required=True)
    p.add_argument("--D", type=int, required=True)
    p.add_argument("--seed", required=True, help="pickle path | bp | FM | FQ | AFQ | SU3 | random")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--chi", type=int, default=None, help="optimization chi (default 2 D^2)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--noise", type=float, default=1e-3)
    p.add_argument("--rng", type=int, default=0)
    a = p.parse_args()

    import yastn.tn.fpeps as fpeps

    from bbh_tri import variational as var
    from bbh_tri.hamiltonian import Model
    from bbh_tri.yastn_compat import peps_to_dict

    theta = np.pi * a.theta_over_pi
    model = Model(theta)
    D, chi = a.D, a.chi or 2 * a.D * a.D
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ops = var.torch_ops()
    ck = a.out + ".pt"
    if os.path.exists(ck):  # resume: the template defines the tensor structure; load() checks theta, chi, D
        import torch
        template = fpeps.Peps.from_dict(torch.load(ck, weights_only=False)["template"], config=ops.config)
        chain = var.Chain(template, model, ops, chi, checkpoint=ck, log=print).load(ck)
        print(f"resumed at step {len(chain.history)}", flush=True)
    else:
        psi = seed_state(a.seed, D, theta, a.noise, a.rng)
        chain = var.Chain(var.convert(psi, ops.config), model, ops, chi, checkpoint=ck, log=print)
    status = chain.run(max_steps=a.max_steps)
    psi = chain.numpy_peps()
    with open(a.out + ".pkl", "wb") as f:
        pickle.dump(peps_to_dict(psi), f)
    val = var.validate(psi, model, [D * D, 2 * D * D, 3 * D * D])
    rec = {"theta_over_pi": a.theta_over_pi, "D": D, "chi_opt": chi, "seed": a.seed, "noise": a.noise,
           "rng": a.rng, "status": status, "energy_opt": chain.history[-1]["energy"] if chain.history else None,
           "validation": {**val, "order": {str(k): v for k, v in val["order"].items()}},
           "history": chain.history}
    with open(a.out + ".json", "w") as f:
        json.dump(rec, f, indent=1)
    print(status, json.dumps(val["energies"]), flush=True)


if __name__ == "__main__":
    main()
