"""M2: D = 1 product states, mean-field energies and the YASTN pipeline (SPEC §3, §6 M2).

Reference values: SPEC §3 (closed forms checked numerically 2026-10-14; Niesen & Corboz,
PRB 97, 245146 (2018) for the FM energy and the mean-field theta_c = arctan(-2) + 2 pi).
"""

import numpy as np
import pytest

from bbh_tri import lattice, measure, operators, states
from bbh_tri.hamiltonian import Model

THETA_MF = np.arctan(-2) + 2 * np.pi  # mean-field FQ -> AFM3 onset (SPEC §3: 1.64758 pi, not 1.658 pi)

TABLE = [  # (theta, E/site of the optimal 3-sublattice product state), SPEC §3
    (0.0, -1.5),
    (np.pi / 4, 3 / np.sqrt(2)),
    (np.pi / 2, 3.0),
    (5 * np.pi / 4, -3 * np.sqrt(2)),
    (THETA_MF, -12 / np.sqrt(5)),
]


@pytest.fixture(scope="module")
def optimized():
    return {th: states.optimize_product_state(Model(th), restarts=6) for th, _ in TABLE}


@pytest.mark.parametrize("theta,e_ref", TABLE)
def test_product_state_table(optimized, theta, e_ref):
    assert optimized[theta][0] == pytest.approx(e_ref, abs=1e-10)


@pytest.mark.parametrize("theta", np.linspace(0, 2 * np.pi, 9))
def test_seed_energy_formulas(theta):
    """E_FM = 3(cos + sin), E_FQ = 6 sin, E_AFQ = 3 sin (SPEC §3)."""
    m = Model(theta)
    assert states.product_energy(m, states.seed_vectors("FM")) == pytest.approx(3 * (np.cos(theta) + np.sin(theta)), abs=1e-12)
    assert states.product_energy(m, states.seed_vectors("FQ")) == pytest.approx(6 * np.sin(theta), abs=1e-12)
    assert states.product_energy(m, states.seed_vectors("AFQ")) == pytest.approx(3 * np.sin(theta), abs=1e-12)


def test_120_coherent_state_is_not_optimal_at_su3_point():
    """SPEC §3: the 120° coherent state gives 3(-cos/2 + 1.5625 sin) = 2.254 at pi/4, above 3/sqrt2."""
    # spin-1 coherent states R_y(phi)|+1> = (cos^2(phi/2), sin(phi)/sqrt2, sin^2(phi/2)), phi = 0, 120°, 240°
    vs = [np.array([np.cos(p / 2) ** 2, np.sin(p) / np.sqrt(2), np.sin(p / 2) ** 2]) for p in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    e = states.product_energy(Model(np.pi / 4), vs)
    assert e == pytest.approx(3 * (-np.cos(np.pi / 4) / 2 + 1.5625 * np.sin(np.pi / 4)), abs=1e-12)
    assert e > 3 / np.sqrt(2) + 0.1
    assert all(states.moment(v) == pytest.approx(1) for v in vs)


@pytest.mark.parametrize("theta", [1.55 * np.pi, 1.6 * np.pi, THETA_MF - 1e-3])
def test_fq_below_mean_field_onset(theta):
    """3 pi/2 < theta <= theta_c^MF: the optimum is FQ, E = 6 sin(theta), m = 0 (SPEC §6 M2)."""
    e, vs = states.optimize_product_state(Model(theta), restarts=4)
    assert e == pytest.approx(6 * np.sin(theta), abs=1e-10)
    assert max(states.moment(v) for v in vs) < 1e-5


def test_mean_field_onset_is_continuous():
    """Above theta_c^MF the moment grows from zero and E_opt - E_FQ ~ (theta - theta_c)^2 (SPEC §3)."""
    data = {}
    for d in (0.004 * np.pi, 0.008 * np.pi):
        e, vs = states.optimize_product_state(Model(THETA_MF + d), restarts=4)
        data[d] = (6 * np.sin(THETA_MF + d) - e, max(states.moment(v) for v in vs))
    (g1, m1), (g2, m2) = data[0.004 * np.pi], data[0.008 * np.pi]
    assert 0 < m1 < m2 < 0.5
    assert g2 / g1 == pytest.approx(4, rel=0.2)


@pytest.mark.parametrize("theta_pi,m_ref", [(1.66, 0.47), (1.7, 0.78), (1.8, 0.95), (1.9, 0.99)])
def test_afm3_reduced_moment(theta_pi, m_ref):
    """Optimal AFM3 product state has a reduced moment (SPEC §3: 0.47, 0.78, 0.95, 0.99)."""
    e, vs = states.optimize_product_state(Model(theta_pi * np.pi), restarts=4)
    ms = [states.moment(v) for v in vs]
    assert all(m == pytest.approx(m_ref, abs=0.01) for m in ms)
    assert e < 6 * np.sin(theta_pi * np.pi)
    if theta_pi == 1.7:
        assert e == pytest.approx(-4.9576, abs=1e-4)  # SPEC §3


# --- full YASTN pipeline (PEPS -> CTM -> rdm / nsite) at D = 1 ---------------------------------

PIPELINE_MODELS = [Model(0.0), Model(np.pi / 4), Model(1.7 * np.pi),
                   Model(0.3, J2=0.4, K2=-0.2), Model(1.2, J2=-0.3, K2=0.5, J_anis=0.6)]


@pytest.mark.parametrize("model", PIPELINE_MODELS, ids=lambda m: f"th{m.theta:.2f}_J2{m.J2}_K2{m.K2}_a{m.J_anis}")
def test_pipeline_matches_mean_field(model):
    """Optimized 3-sublattice state on the sqrt3 cell: CTM energy per class == product formula, all classes."""
    ops = operators.spin1("dense", default_dtype="complex128")
    e_mf, vs = states.optimize_product_state(model, restarts=2)
    psi = states.product_peps(lattice.geometry("sqrt3"), vs, ops)
    env = measure.ctm_env(psi, chi=1)
    e, by_class = measure.energy(psi, env, ops, model)
    assert set(by_class) == set(model.couplings())
    assert e == pytest.approx(e_mf, abs=1e-10)
    for site, (m, q) in measure.order_parameters(psi, env, ops).items():
        assert m == pytest.approx(states.moment(vs[lattice.sublattice(site)]), abs=1e-10)


@pytest.mark.parametrize("kind", ["FM", "FQ", "SU3"])
@pytest.mark.parametrize("model", [Model(0.4), Model(2.0, J2=0.3, K2=0.1)], ids=["nn", "nnn"])
def test_pipeline_u1_matches_dense(kind, model):
    """U1-compatible seeds (Sz eigenstates) give the same energies and |Q| in U1 and dense."""
    out = {}
    for sym in ("dense", "U1"):
        ops = operators.spin1(sym)
        psi = states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors(kind), ops)
        env = measure.ctm_env(psi, chi=1)
        out[sym] = (measure.energy(psi, env, ops, model), measure.order_parameters(psi, env, ops))
    (e_d, c_d), op_d = out["dense"]
    (e_u, c_u), op_u = out["U1"]
    assert e_u == pytest.approx(e_d, abs=1e-12)
    assert e_d == pytest.approx(states.product_energy(model, states.seed_vectors(kind)), abs=1e-12)
    assert all(c_u[k] == pytest.approx(c_d[k], abs=1e-12) for k in c_d)
    for s in op_d:
        assert op_u[s][0] == pytest.approx(op_d[s][0], abs=1e-12)
        assert op_u[s][1] == pytest.approx(op_d[s][1], abs=1e-12)


def test_q_norm_through_pipeline():
    """|Q| = 2/sqrt3 for |0> and 1/sqrt3 for |+-1> (SPEC §3), measured through CTM."""
    ops = operators.spin1("dense")
    psi = states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors("SU3"), ops)
    env = measure.ctm_env(psi, chi=1)
    for site, (m, q) in measure.order_parameters(psi, env, ops).items():
        sz = (1, 0, -1)[lattice.sublattice(site)]
        assert q == pytest.approx(2 / np.sqrt(3) if sz == 0 else 1 / np.sqrt(3), abs=1e-12)
        assert m == pytest.approx(abs(sz), abs=1e-12)


@pytest.mark.parametrize("dims", [(2, 3), (3, 3)])
def test_obc_patch_energy_matches_bond_sum(dims):
    """Finite OBC patch: per-class means over the patch bonds equal the product-state bond energies."""
    ops = operators.spin1("dense")
    model = Model(0.5, J2=0.3, K2=0.2)
    vs = states.seed_vectors("SU3")
    geo = lattice.geometry("obc", dims)
    psi = states.product_peps(geo, vs, ops)
    env = measure.ctm_env(psi, chi=1)
    _, by_class = measure.energy(psi, env, ops, model)
    assert set(by_class) == {c for c in model.couplings() if lattice.bonds(geo, c)}
    for cls, (J, K) in model.couplings().items():
        if cls not in by_class:
            continue
        H = operators.as_matrix(ops, operators.h_bond(ops, J, K))
        ref = np.mean([np.kron(vs[lattice.sublattice(a)], vs[lattice.sublattice(b)]) @ H @
                       np.kron(vs[lattice.sublattice(a)], vs[lattice.sublattice(b)]) for a, b in lattice.bonds(geo, cls)])
        assert by_class[cls] == pytest.approx(ref, abs=1e-12), cls
