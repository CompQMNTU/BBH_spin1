"""Test-only helpers for the M0 YASTN checks (ported from the M0 scratch scripts).

Physics code proper (operators, Hamiltonians, lattice) belongs in src/bbh_tri from M1 on.
"""

import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site
from yastn.tn.fpeps.envs.rdm import rdm1x2, rdm2x1, rdm2x2_anti_diagonal


def spin1(sym="dense"):
    return yastn.operators.Spin1(sym=sym)


def ss_numpy(ops):
    """S·S on two spin-1 sites as a 9x9 matrix, in the basis order of `ops`."""
    Sz, Sp, Sm = ops.sz().to_numpy(), ops.sp().to_numpy(), ops.sm().to_numpy()
    return np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp))


def ss_yastn(ops):
    Sz, Sp, Sm = ops.sz(), ops.sp(), ops.sm()
    return yastn.fkron(Sz, Sz) + 0.5 * (yastn.fkron(Sp, Sm) + yastn.fkron(Sm, Sp))


def bbh_bond_numpy(ops, theta):
    SS = ss_numpy(ops)
    return np.cos(theta) * SS + np.sin(theta) * SS @ SS


def bbh_bond_yastn(ops, theta):
    SS = ss_yastn(ops)
    SSf = SS.fuse_legs(axes=((0, 2), (1, 3)))
    SS2 = (SSf @ SSf).unfuse_legs(axes=(0, 1)).transpose(axes=(0, 2, 1, 3))
    return np.cos(theta) * SS + np.sin(theta) * SS2


def diag_path(bond):
    """3-site square-NN path for the anti-diagonal bond ((x+1,y),(x,y+1)), via (x,y)."""
    s0, s1 = bond
    return (s0, Site(s1[0], s0[1]), s1)


def peps_vector(psi, Nx, Ny):
    """Dense state vector of a finite PEPS; returns (array with one leg per site, site order)."""
    t = psi.to_tensor().to_numpy()
    t = t.reshape([3] * (Nx * Ny))
    # to_tensor leg order is column-major in (nx, ny)
    order = [Site(nx, ny) for ny in range(Ny) for nx in range(Nx)]
    return t, order


def apply_2site(vec, order, U, sa, sb):
    """Apply a 9x9 operator U on sites (sa, sb) of a dense state vector."""
    ia, ib = order.index(sa), order.index(sb)
    out = np.tensordot(U.reshape(3, 3, 3, 3), vec, axes=((2, 3), (ia, ib)))
    rest = [k for k in range(vec.ndim) if k not in (ia, ib)]
    return out.transpose(np.argsort([ia, ib] + rest))


def energy_per_class(psi, env, ops, theta):
    """NN bond energies per class (h, v, d) of the BBH model from a converged EnvCTM."""
    h4 = bbh_bond_numpy(ops, theta).reshape(3, 3, 3, 3)

    def ev(rdm):
        r = rdm.to_numpy()  # legs (s0, s0', s1, s1')
        return float(np.real(np.einsum("ikjl,jilk->", h4, r) / np.einsum("iikk->", r)))

    geo = psi.geometry
    E = {
        "h": [ev(rdm1x2(b[0], psi, env)[0]) for b in geo.bonds("h")],
        "v": [ev(rdm2x1(b[0], psi, env)[0]) for b in geo.bonds("v")],
        # b = ((x+1,y),(x,y+1)); the anti-diagonal 2x2 patch is anchored at (x,y)
        "d": [ev(rdm2x2_anti_diagonal(Site(b[1][0], b[0][1]), psi, env)[0]) for b in geo.bonds("d")],
    }
    e_site = sum(sum(v) for v in E.values()) / len(geo.sites())
    return e_site, {k: float(np.mean(v)) for k, v in E.items()}


def converged_ctm(psi, chi, max_sweeps=200, corner_tol=1e-11):
    env = fpeps.EnvCTM(psi, init="eye")
    env.iterate_(opts_svd={"D_total": chi}, moves="hv", method="2x2",
                 max_sweeps=max_sweeps, corner_tol=corner_tol)
    return env
