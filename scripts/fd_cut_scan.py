"""Loss and CTM cut along fd_check's random directions at a Chain checkpoint (issue #26).

Forward-only, numpy backend (torch on a CPU took > 25 min per D = 5 evaluation, numpy 8.5 min
cold): the chain's gap-safe cut, CTM to corner_tol, energy; each point's CTM starts from the
converged environment at t = 0 (``--cold`` to start every point from scratch). For each of fd_check's random directions (seed = the gate's step, as Chain.check_gradient
uses) it evaluates the loss and ``variational.cut_signature`` at t = k eps / 16, k in --ks,
eps = rel_eps |p|, and prints the central differences at eps, eps/4, eps/16 with whether the
stencil stays on the cut of t = 0. A loss jump where the signature changes, with smooth pieces in
between, confirms the cut-jump reading of an FD outlier.

  uv run --extra ad python scripts/fd_cut_scan.py CHECKPOINT.pt --chi 74 --seed 321 [--theta-over-pi 0]
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True, help="fd_check seed (Chain uses the step number)")
    ap.add_argument("--theta-over-pi", type=float, default=0.0)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--rel-eps", type=float, default=1e-4)
    ap.add_argument("--ks", default="1,2,4,8,12,16", help="positive k of t = +-k eps / 16")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", help="JSON lines output (default: CHECKPOINT.fdscan.jsonl)")
    a = ap.parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(a.threads)

    import numpy as np
    import torch
    import yastn.tn.fpeps as fpeps
    from bbh_tri import measure, operators, variational as var
    from bbh_tri.hamiltonian import Model
    torch.set_num_threads(a.threads)

    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    ops = var.torch_ops()
    template = fpeps.Peps.from_dict(saved["template"], config=ops.config)
    ad = saved["meta"]["ad"]
    ch = var.Chain(template, Model(a.theta_over_pi * np.pi), ops, chi=a.chi, ad=ad).load(a.checkpoint)
    nops, model = operators.spin1("dense"), Model(a.theta_over_pi * np.pi)
    cut = ch.truncation_f  # the chain's gap-safe mask (svd_tol, chi, eps_multiplet as saved)
    params = ch.params
    x0 = [p.detach().clone() for p in params]
    eps = a.rel_eps * float(torch.sqrt(sum((p.abs() ** 2).sum() for p in x0)))
    gen = torch.Generator().manual_seed(a.seed)  # as fd_check: the gradient direction draws nothing
    dirs = []
    for _ in range(a.n_random):
        v = [torch.randn(p.shape, generator=gen, dtype=p.dtype) for p in params]
        n = torch.sqrt(sum((x.abs() ** 2).sum() for x in v))
        dirs.append([x / n for x in v])

    out = open(a.out or a.checkpoint.replace(".pt", "") + ".fdscan.jsonl", "a")

    env0 = None

    def at(t, v):
        nonlocal env0
        with torch.no_grad():
            for p, q, x in zip(params, x0, v):
                p.copy_(q + t * x)
        psi = ch.numpy_peps()
        t0 = time.time()
        env = fpeps.EnvCTM(psi, init="eye" if env0 is None or a.cold else None)
        if env0 is not None and not a.cold:
            env.env, env.proj = env0.env.copy(), env0.proj.copy()
        info = env.iterate_(opts_svd=dict(ch.ctm_fwd["opts_svd"]), moves="hv", method="2x2", max_sweeps=200,
                            corner_tol=ch.ctm_fwd["corner_tol"], truncation_f=cut)
        if not info.converged:
            raise RuntimeError(f"CTM not converged at t = {t}: {info}")
        if env0 is None:
            env0 = env
        E = measure.energy(psi, env, nops, model)[0]
        return E, var.cut_signature(env), time.time() - t0, info.sweeps

    print(f"eps = {eps:.4e}", flush=True)
    ks = sorted(int(k) for k in a.ks.split(","))
    grid = [-k for k in reversed(ks)] + [0] + ks
    base = at(0.0, dirs[0])
    print(f"t=0: E={base[0]:.13f} ({base[2]:.0f}s, {base[3]} sweeps)", flush=True)
    for j, v in enumerate(dirs):
        vals = {0: base}
        for k in grid:
            if k:
                vals[k] = at(k * eps / 16, v)
            E, sig, dt, sw = vals[k]
            changed = [[list(s0), d, [list(D) for _, D in legs]] for (s0, d, legs), (_, _, l0) in zip(sig, base[1]) if legs != l0]
            rec = {"direction": j, "k": k, "t": k * eps / 16, "E": E, "same_cut": sig == base[1],
                   "changed": changed, "time": dt, "sweeps": sw}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"dir {j} t={k:+d}/16 eps  E={E:.13f}  {'same cut' if sig == base[1] else 'CUT CHANGED ' + str(changed)}"
                  f"  ({dt:.0f}s, {sw} sweeps)", flush=True)
        for k in (16, 4, 1):
            if k in vals and -k in vals:
                fd = (vals[k][0] - vals[-k][0]) / (2 * k * eps / 16)
                same = vals[k][1] == base[1] and vals[-k][1] == base[1]
                print(f"dir {j}: FD at {k}/16 eps = {fd:+.4e}  stencil on the t=0 cut: {same}", flush=True)
    with torch.no_grad():
        for p, q in zip(params, x0):
            p.copy_(q)


if __name__ == "__main__":
    main()
