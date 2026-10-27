"""Triangle-MPO gates (SPEC §4.2 option A): operator identity, bond coverage, action on a finite PEPS."""

import numpy as np
import pytest
import scipy.linalg as sl
import yastn
import yastn.tn.fpeps as fpeps

from bbh_tri import gates, lattice, operators, states
from bbh_tri.hamiltonian import Model
from yastn_utils import peps_vector

TAU = 0.2
I3 = np.eye(3)


def triangle_matrix(h_ab, h_bc, h_ac):
    """27x27 h_ab ⊗ 1 + 1 ⊗ h_bc + h_ac (identity on b), canonical basis, site order (a, b, c)."""
    ac = np.kron(h_ac, I3).reshape([3] * 6).transpose(0, 2, 1, 3, 5, 4).reshape(27, 27)
    return np.kron(h_ab, I3) + np.kron(I3, h_bc) + ac


def mpo_matrix(ops, G):
    """Contract [G_a, G_b, G_c] to a 27x27 matrix in the canonical basis."""
    T = yastn.ncon(G, ([-1, -2, 1], [-3, -4, 1, 2], [-5, -6, 2]))
    leg = ops.space()
    a = T.to_numpy(legs={i: leg if s == leg.s else leg.conj() for i, s in enumerate(T.s)})
    p = operators.basis_permutation(ops)
    return a[np.ix_(p, p, p, p, p, p)].transpose(0, 2, 4, 1, 3, 5).reshape(27, 27)


@pytest.mark.parametrize("sym", ["dense", "U1"])
def test_triangle_mpo_reconstructs_exp(sym):
    """SPEC §5 test_gates: the 3-site MPO equals exp(-tau h_tri) to 1e-12, with distinct bond terms."""
    ops = operators.spin1(sym)
    hv, hh, hd = (operators.h_bond(ops, J, K) for J, K in [(0.4, 0.9), (np.cos(0.7), np.sin(0.7)), (-0.3, 0.5)])
    G = gates.exp_mpo(TAU, gates.triangle_hamiltonian(ops, hv, hh, hd))
    ref = sl.expm(-TAU * triangle_matrix(*(operators.as_matrix(ops, h) for h in (hv, hh, hd))))
    assert np.abs(mpo_matrix(ops, G) - ref).max() < 1e-12


def test_infinite_lattice_one_triangle_per_site_covers_each_bond_once():
    geo = lattice.geometry("sqrt3")
    gl = gates.nn_gates(geo, Model(0.3), operators.spin1("dense"), TAU)
    assert len(gl) == len(geo.sites()) and all(len(g.G) == 3 for g in gl)
    seen = {c: [] for c in lattice.NN_CLASSES}
    for g in gl:
        a, b, c = g.sites
        for cls, (s0, s1) in {"v": (b, a), "h": (b, c), "d": (a, c)}.items():
            assert (s1[0] - s0[0], s1[1] - s0[1]) == lattice.NN[cls]
            seen[cls].append(lattice.sublattice(s0))
    # each class: one bond per unit-cell site, i.e. every sublattice once as the bond's first site
    assert all(sorted(v) == [0, 1, 2] for v in seen.values())


def exact_sequence(v, order, gl, hs, swap_h_d=False):
    """Apply the exact operator of every gate in ``gl`` to the canonical-basis vector ``v``."""
    def apply(vec, U, sites):
        idx = [order.index(s) for s in sites]
        k = len(sites)
        out = np.tensordot(U.reshape([3] * (2 * k)), vec, axes=(list(range(k, 2 * k)), idx))
        return out.transpose(np.argsort(idx + [i for i in range(vec.ndim) if i not in idx]))

    for g in gl:
        if len(g.G) == 3:
            hh, hd = (hs["d"], hs["h"]) if swap_h_d else (hs["h"], hs["d"])
            v = apply(v, sl.expm(-TAU * triangle_matrix(hs["v"], hh, hd)), g.sites)
        else:  # a 2-site gate, possibly routed: it acts on the path's end sites only
            s0, s1 = g.sites[0], g.sites[-1]
            cls = {vec: c for c, vec in lattice.VECTORS.items()}[(s1[0] - s0[0], s1[1] - s0[1])]
            v = apply(v, sl.expm(-TAU * hs[cls]), (s0, s1))
    return v


NNN_MODEL = Model(0.7, J2=0.4, K2=-0.3, J_anis=0.6)


@pytest.mark.parametrize("dims,model,variant", [
    ((2, 3), Model(0.7, J_anis=0.6), 0), ((3, 3), Model(0.7, J_anis=0.6), 0),
    # 2x3 holds 'nnn_d' and 'nnn_k1', 3x2 'nnn_d' and 'nnn_k2'. Not 3x3: untruncated knight gates
    # through 9 sites need arrays of 13-20 GB
    *[(dims, NNN_MODEL, v) for dims in [(2, 3), (3, 2)] for v in range(3)]])
def test_gates_on_obc_patch_match_exact(dims, model, variant):
    """One first-order sweep without truncation equals the exact gate sequence (all bonds, once).

    J_anis != 1 makes the bond classes distinguishable; putting the h and d terms on the wrong
    triangle bonds gives a visibly different state. With J2, K2 (M5 acceptance (a)) every NNN
    class is routed along path ``variant``.
    """
    ops = operators.spin1("dense")
    geo = lattice.geometry("obc", dims)
    psi = states.random_product_peps(geo, ops, seed=3)
    gl = gates.model_gates(geo, model, ops, TAU, variant)
    n_bonds = sum(3 if len(g.G) == 3 else 1 for g in gl)
    assert n_bonds == sum(len(lattice.bonds(geo, c)) for c in model.couplings())

    p = operators.basis_permutation(ops)
    v0, order = peps_vector(psi, *dims)
    v0 = v0[np.ix_(*[p] * v0.ndim)]
    hs = {c: operators.as_matrix(ops, h) for c, h in model.bond_hamiltonians(ops).items()}
    fpeps.evolution_step_(fpeps.EnvBP(psi), gl, opts_svd={"D_total": 729, "tol": 1e-15})
    v1, _ = peps_vector(psi, *dims)
    v1 = v1[np.ix_(*[p] * v1.ndim)]

    def overlap(w):
        return abs(np.vdot(v1, w)) / np.linalg.norm(v1) / np.linalg.norm(w)

    assert overlap(exact_sequence(v0, order, gl, hs)) == pytest.approx(1, abs=1e-12)
    assert overlap(exact_sequence(v0, order, gl, hs, swap_h_d=True)) < 1 - 1e-4


def test_trotter2_appends_reverse():
    gl = gates.nn_gates(lattice.geometry("sqrt3"), Model(0.3), operators.spin1("dense"), TAU / 2)
    t2 = gates.trotter2(gl)
    assert [g.sites for g in t2] == [g.sites for g in gl] + [g.sites for g in gl[::-1]]


def test_nnn_gates_need_a_patch_cell():
    """On sqrt3 every NNN pair shares a tensor, so no NNN gate path exists (lattice.gate_paths)."""
    ops = operators.spin1("dense")
    with pytest.raises(ValueError, match="patch cell"):
        gates.model_gates(lattice.geometry("sqrt3"), Model(0.3, J2=0.1), ops, TAU)
    assert len(gates.model_gates(lattice.geometry("sqrt3"), Model(0.3), ops, TAU)) == 3  # NN only: unchanged


@pytest.mark.parametrize("variant", range(6))
def test_model_gates_cover_each_nnn_bond_once(variant):
    """(3, 3) cell: 9 triangles, then one routed 2-site gate per NNN bond (27), each along the
    path ``variant`` picks cyclically."""
    geo = lattice.geometry("patch", (3, 3))
    gl = gates.model_gates(geo, Model(0.3, J2=0.2, K2=-0.1), operators.spin1("dense"), TAU, variant)
    tri, nnn = gl[:9], gl[9:]
    assert all(len(g.G) == 3 for g in tri) and len(nnn) == 27 and all(len(g.G) == 2 for g in nnn)
    expected = [lattice.gate_paths(geo, b)[variant % len(lattice.gate_paths(geo, b))]
                for cls in lattice.NNN_CLASSES for b in lattice.bonds(geo, cls)]
    assert [g.sites for g in nnn] == expected
    assert gates.model_gates(geo, Model(0.3, K2=0.2), operators.spin1("dense"), TAU, variant)[9:]  # K2 alone counts
