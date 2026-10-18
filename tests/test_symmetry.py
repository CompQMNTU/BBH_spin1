"""M4 symmetry switch (SPEC §4.5): U1, U1xU1 and Z2xZ2 against dense.

SPEC §4.5 test: each symmetric implementation reproduces the dense energy of its compatible product
state (D = 1) and agrees with dense at small D when started from the same state. The second holds
where the dense run keeps the symmetry by itself: U1 from FQ and from SU3, Z2xZ2 from AFQ at D = 2,
U1xU1 from SU3 at D = 3 (1e-11). At D = 2 the dense BP run from the SU3 seed breaks U1xU1
(symmetry-violating rdm elements ~1e-3) and ends lower (1.8401 vs 1.8774); there the test checks
only the symmetric run's self-consistency (its dense copy has the same energy).
"""

import numpy as np
import pytest
import scipy.linalg

from bbh_tri import evolve, lattice, measure, operators, states
from bbh_tri.hamiltonian import Model

DENSE = operators.spin1("dense")
GEO = lattice.geometry("sqrt3")

# (symmetry, theta / pi, compatible product seeds)
CASES = [("U1", 1.6, ("FM", "FQ", "SU3")), ("U1xU1", 0.25, ("SU3", "FM", "FQ")), ("Z2xZ2", 0.35, ("AFQ", "FQ"))]


def matrix(ops, op):
    return operators.as_matrix(ops, op)


@pytest.mark.parametrize("sym", ["U1", "U1xU1", "Z2xZ2"])
def test_bond_hamiltonian_matches_dense(sym):
    ops = operators.spin1(sym)
    J = K = 0.6  # SU(3) point: the only (J, K) direction U1xU1 allows
    assert np.allclose(matrix(ops, operators.h_bond(ops, J, K)), matrix(DENSE, operators.h_bond(DENSE, J, K)), atol=1e-14)
    if sym != "U1xU1":
        assert np.allclose(matrix(ops, operators.h_bond(ops, 0.3, -0.8)),
                           matrix(DENSE, operators.h_bond(DENSE, 0.3, -0.8)), atol=1e-14)


def test_u1xu1_rejects_non_su3_couplings():
    with pytest.raises(ValueError, match="mixes charges"):
        operators.h_bond(operators.spin1("U1xU1"), 0.3, 0.7)


@pytest.mark.parametrize("sym, S, Q", [("U1xU1", {"Sz"}, {"Q2"}), ("Z2xZ2", set(), {"Q1", "Q2"})])
def test_order_parameter_components_are_the_neutral_ones(sym, S, Q):
    ops = operators.spin1(sym)
    assert set(operators.spin_vector(ops)) == S and set(operators.quadrupole(ops)) == Q
    for k, op in {**operators.spin_vector(ops), **operators.quadrupole(ops)}.items():
        dense = {**operators.spin_vector(DENSE), **operators.quadrupole(DENSE)}[k]
        assert np.allclose(matrix(ops, op), matrix(DENSE, dense), atol=1e-14)


def test_z2xz2_charges_are_pi_rotation_parities():
    """exp(i pi S_a) acts on config basis state k as (-1)^charge."""
    ops = operators.spin1("Z2xZ2")
    B = operators.basis_matrix(ops)
    for g, axis in ((0, "sx"), (1, "sz")):
        Sa = matrix(DENSE, getattr(DENSE, axis)())
        R = B.conj().T @ scipy.linalg.expm(1j * np.pi * Sa) @ B
        assert np.allclose(R, np.diag([(-1) ** q[g] for q in ops.charges]), atol=1e-14)


@pytest.mark.parametrize("sym, t, seeds", CASES)
def test_product_state_energy_matches_dense(sym, t, seeds):
    ops, model = operators.spin1(sym), Model(np.pi * t)
    for seed in seeds:
        psi = states.product_peps(GEO, states.seed_vectors(seed), ops)
        E = measure.energy(psi, measure.ctm_env(psi, 4), ops, model)[0]
        assert E == pytest.approx(states.product_energy(model, states.seed_vectors(seed)), abs=1e-12)
        dense = states.to_dense(psi, ops)
        assert measure.energy(dense, measure.ctm_env(dense, 4), DENSE, model)[0] == pytest.approx(E, abs=1e-12)


def test_incompatible_product_state_raises():
    with pytest.raises(ValueError, match="charge eigenstate"):
        states.product_peps(GEO, states.seed_vectors("AFQ"), operators.spin1("U1xU1"))


def bp(sym, seed, t, D):
    ops, model = operators.spin1(sym), Model(np.pi * t)
    psi = states.product_peps(GEO, states.seed_vectors(seed), ops)
    E = evolve.evolve(psi, model, ops, D, schedule=(0.1, 0.05), max_steps=100).energy
    return psi, ops, model, E


@pytest.mark.slow
@pytest.mark.parametrize("sym, seed, t, tol", [("U1", "FQ", 1.6, 1e-6), ("U1", "SU3", 0.25, 1e-10),
                                               ("Z2xZ2", "AFQ", 0.35, 1e-9)])
def test_bp_d2_matches_dense_from_same_seed(sym, seed, t, tol):
    psi, ops, model, E = bp(sym, seed, t, 2)
    assert E == pytest.approx(bp("dense", seed, t, 2)[3], abs=tol)
    dense = states.to_dense(psi, ops)
    assert measure.energy(dense, measure.ctm_env(dense, 4), DENSE, model)[0] == pytest.approx(E, abs=1e-10)


@pytest.mark.slow
def test_u1xu1_bp_d3_matches_dense_from_su3_seed():
    assert bp("U1xU1", "SU3", 0.25, 3)[3] == pytest.approx(bp("dense", "SU3", 0.25, 3)[3], abs=1e-9)


@pytest.mark.slow
def test_u1xu1_bp_d2_is_self_consistent_and_above_dense():
    psi, ops, model, E = bp("U1xU1", "SU3", 0.25, 2)
    dense = states.to_dense(psi, ops)
    assert measure.energy(dense, measure.ctm_env(dense, 4), DENSE, model)[0] == pytest.approx(E, abs=1e-10)
    assert E > bp("dense", "SU3", 0.25, 2)[3]  # the dense run breaks U1xU1 (module docstring)


@pytest.mark.slow
@pytest.mark.ad
@pytest.mark.parametrize("sym, seed, t", [("U1xU1", "SU3", 0.25), ("Z2xZ2", "AFQ", 0.35)])
def test_symmetric_ad_gradient_matches_fd(sym, seed, t):
    """AD through CTM on block-sparse tensors: directional derivative vs central differences."""
    torch = pytest.importorskip("torch")
    from bbh_tri import variational as var
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    import warnings

    ops, model = operators.spin1(sym), Model(np.pi * t)
    psi = states.product_peps(GEO, states.seed_vectors(seed), ops)
    evolve.evolve(psi, model, ops, 2, schedule=(0.1,), max_steps=20)
    tops = var.torch_ops(sym)
    ch = var.Chain(var.convert(psi, tops.config), model, tops, chi=8)
    with exact_svd_backward("zero"), warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateSVDWarning)
        E, _ = ch.loss()
        E.backward()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in ch.params))
    v = [p.grad / gnorm for p in ch.params]  # steepest direction: a random one gives ~1e-5, below FD noise
    ad = gnorm.item()
    x0 = [p.detach().clone() for p in ch.params]
    vals = []
    for sgn in (1, -1):
        with torch.no_grad():
            for p, q, vi in zip(ch.params, x0, v):
                p.copy_(q + sgn * 1e-5 * vi)
        vals.append(ch.loss()[0].item())
    assert ad == pytest.approx((vals[0] - vals[1]) / 2e-5, rel=1e-5)


@pytest.mark.parametrize("sym, seed, t, J2, K2", [("U1xU1", "SU3", 0.25, 0.3, 0.3), ("Z2xZ2", "AFQ", 0.35, 0.3, -0.2)])
def test_knight_nnn_energy_matches_dense(sym, seed, t, J2, K2):
    """Knight-move NNN classes have no rdm; SymSpin1 measures them via operators.two_site_terms."""
    ops, model = operators.spin1(sym), Model(np.pi * t, J2=J2, K2=K2)
    psi = states.product_peps(GEO, states.seed_vectors(seed), ops)
    env = measure.ctm_env(psi, 4)
    E, by_class = measure.energy(psi, env, ops, model)
    knight = [c for c in by_class if c not in measure.RDM_CLASSES]
    assert knight  # the test exercises the no-rdm path
    assert E == pytest.approx(states.product_energy(model, states.seed_vectors(seed)), abs=1e-12)
    dense = states.to_dense(psi, ops)
    Ed, by_d = measure.energy(dense, measure.ctm_env(dense, 4), DENSE, model)
    assert all(by_class[c] == pytest.approx(by_d[c], abs=1e-12) for c in knight)


def test_two_site_terms_recomposes_and_rejects_yastn_spin1():
    ops = operators.spin1("Z2xZ2")
    h = operators.h_bond(ops, 0.3, -0.8)
    total = sum(c * np.kron(operators.as_matrix(ops, A), operators.as_matrix(ops, B))
                for c, A, B in operators.two_site_terms(ops, h))
    assert np.allclose(total, operators.as_matrix(ops, h), atol=1e-13)
    with pytest.raises(TypeError):
        operators.two_site_terms(DENSE, operators.h_bond(DENSE, 0.3, -0.8))
