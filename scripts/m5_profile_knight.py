"""U3 (docs/yastn_findings.md): cost of the NNN classes (knight windows via measure_nsite_exact)
at D = 3-5, chi = 3D^2, sqrt3 cell, one thread. Needs --extra ad; D = 4 AD with NNN peaks ~37 GB.

    OMP_NUM_THREADS=1 uv run --extra ad python scripts/m5_profile_knight.py
"""
import resource
import time

import numpy as np

from bbh_tri import evolve, lattice, measure, operators, states, variational as var
from bbh_tri.hamiltonian import Model
from bbh_tri.yastn_compat import exact_svd_backward, gap_safe_truncation
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
ops = operators.spin1("dense")
model = Model(0.25 * np.pi, J2=0.2, K2=0.1)
for D in (3, 4, 5):
    psi = states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors("SU3"), ops)
    var_psi = var.pad(psi, D, noise=1e-2, seed=0)
    evolve.evolve(var_psi, Model(0.25 * np.pi), ops, D, schedule=(0.05,), max_steps=40, check_every=40)
    chi = 3 * D * D
    t = time.time(); env = measure.ctm_env(var_psi, chi, corner_tol=1e-9, truncation_f=gap_safe_truncation(1e-8, chi, eps_multiplet=1e-6))
    t_ctm = time.time() - t
    times = {}
    for cls, (J, K) in model.couplings().items():
        b = lattice.bonds(var_psi.geometry, cls)
        t = time.time(); [measure.bond_energy(var_psi, env, ops, bd, cls, J, K) for bd in b]; times[cls] = time.time() - t
    print(f"numpy D={D} chi={chi}: CTM {t_ctm:.1f}s; per class (3 bonds each) " + ", ".join(f"{c} {v:.2f}s" for c, v in times.items()) + f"; maxrss {rss():.1f} GB", flush=True)
    if D <= 4:
        tops = var.torch_ops()
        for m in (Model(0.25 * np.pi), model):
            ch = var.Chain(var.convert(var_psi, tops.config), m, tops, chi=chi, ad="fp")
            with exact_svd_backward("zero"):
                t = time.time(); E, _ = ch.loss(); tf = time.time() - t
                t = time.time(); E.backward(); tb = time.time() - t
            print(f"fp AD D={D} chi={chi} {'NNN' if m.has_nnn else 'NN '}: forward {tf:.1f}s backward {tb:.1f}s maxrss {rss():.1f} GB", flush=True)
