"""Measurements on a PEPS with a converged CTM environment (SPEC §2.2, §4.3).

Bond energies per class use reduced density matrices where YASTN has one ('h' rdm1x2, 'v' rdm2x1,
'd' rdm2x2_anti_diagonal, 'nnn_d' rdm2x2_diagonal). Knight-move NNN classes have none; they use
``measure_nsite_exact`` on the operator expansion of h = J S·S + K (S·S)^2 into single-site
products. Never use ``measure_nn()`` without a bond or ``geometry.bonds()`` (SPEC §1).

A two-site operator O (legs out1, in1, out2, in2) is evaluated against an rdm (legs s0, s0', s1,
s1') as tensordot(O, rdm, axes=((0, 1, 2, 3), (1, 0, 3, 2))) / Tr rdm; checked against
``measure_nn`` on a complex state with a non-Hermitian operator.
"""

import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site
from yastn.tn.fpeps.envs.rdm import rdm1x2, rdm2x1, rdm2x2_anti_diagonal, rdm2x2_diagonal

from . import lattice, operators
from .hamiltonian import energy_per_site

RDM_CLASSES = ("h", "v", "d", "nnn_d")


def ctm_env(psi, chi, corner_tol=1e-12, max_sweeps=200):
    """CTM environment (method '2x2', h/v moves) iterated to corner convergence (SPEC §4.3)."""
    env = fpeps.EnvCTM(psi, init="eye")
    env.iterate_(opts_svd={"D_total": chi}, moves="hv", method="2x2", max_sweeps=max_sweeps, corner_tol=corner_tol)
    return env


def bond_rdm(psi, env, bond, cls):
    """Two-site rdm with legs (bond[0] pair, bond[1] pair) for an rdm-supported class."""
    s0, s1 = bond
    if cls == "h":
        return rdm1x2(s0, psi, env)[0]
    if cls == "v":
        return rdm2x1(s0, psi, env)[0]
    if cls == "d":  # s0 bottom-left, s1 = s0 + (-1, 1) top-right; anchor is the upper-left site
        return rdm2x2_anti_diagonal(Site(s1[0], s0[1]), psi, env)[0]
    if cls == "nnn_d":  # s0 upper-left, s1 = s0 + (1, 1)
        return rdm2x2_diagonal(s0, psi, env)[0]
    raise ValueError(f"no rdm for class {cls!r}")


def rdm_expectation(op2, rdm):
    """<O> of a two-site operator from a two-site rdm."""
    num = yastn.tensordot(op2, rdm, axes=((0, 1, 2, 3), (1, 0, 3, 2))).to_number()
    return num / yastn.trace(rdm, axes=((0, 2), (1, 3))).to_number()


def bbh_terms(ops, J, K):
    """[(coefficient, A, B)] with J S·S + K (S·S)^2 = sum coefficient A ⊗ B (single-site A, B).

    S·S = sum_a c_a X_a ⊗ Y_a with (c, X, Y) in {(1, Sz, Sz), (1/2, S+, S-), (1/2, S-, S+)}, so
    (S·S)^2 = sum_ab c_a c_b (X_a X_b) ⊗ (Y_a Y_b).
    """
    Sz, Sp, Sm = ops.sz(), ops.sp(), ops.sm()
    ss = [(1.0, Sz, Sz), (0.5, Sp, Sm), (0.5, Sm, Sp)]
    out = [(J * c, X, Y) for c, X, Y in ss] if J != 0 else []
    if K != 0:
        out += [(K * ca * cb, Xa @ Xb, Ya @ Yb) for ca, Xa, Ya in ss for cb, Xb, Yb in ss]
    return out


def bond_energy(psi, env, ops, bond, cls, J, K):
    """<J S·S + K (S·S)^2> on one bond."""
    if cls in RDM_CLASSES:
        return rdm_expectation(operators.h_bond(ops, J, K), bond_rdm(psi, env, bond, cls))
    return sum(c * env.measure_nsite_exact(A, B, sites=list(bond)) for c, A, B in bbh_terms(ops, J, K))


def class_energies(psi, env, ops, model):
    """{class: mean bond energy} over the bonds of ``psi.geometry`` for every coupled class.

    A class with no bonds on a finite patch (e.g. 'nnn_k2' on 2 rows) is left out.
    """
    out = {}
    for cls, (J, K) in model.couplings().items():
        bs = lattice.bonds(psi.geometry, cls)
        if not bs:
            continue
        out[cls] = sum(bond_energy(psi, env, ops, b, cls, J, K) for b in bs) / len(bs)
    return out


def energy(psi, env, ops, model):
    """(E/site, {class: mean bond energy}); real parts."""
    e = {c: float(np.real(v)) for c, v in class_energies(psi, env, ops, model).items()}
    return energy_per_site(e), e


def site_expectations(env, ops_dict):
    """{name: {site: <op>}} for single-site operators."""
    return {k: env.measure_1site(o) for k, o in ops_dict.items()}


def order_parameters(psi, env, ops):
    """{site: (m, |Q|)}: moment |<S>| and quadrupole norm |<Q>| per site (SPEC §2.2).

    In U1 only Sz and Q2 enter; the other components vanish by symmetry.
    """
    S = site_expectations(env, operators.spin_vector(ops))
    Q = site_expectations(env, operators.quadrupole(ops))
    return {s: (operators.vector_norm({k: v[s] for k, v in S.items()}),
                operators.vector_norm({k: v[s] for k, v in Q.items()})) for s in psi.geometry.sites()}
