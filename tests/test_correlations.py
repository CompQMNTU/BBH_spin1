"""M6 observables (SPEC §4.3): order at Γ/K/M, line correlations, scalar chirality, correlation lengths.

Product-state references are exact (SPEC §3: |Q| of |0> = 2/√3, of |±1> = 1/√3). The tetrahedral
spin-coherent state on (2, 2) has n_i·(n_j × n_k) = 4/(3√3) on every triangle for the vertex
directions (1, 1, 1)/√3, (1, -1, -1)/√3, (-1, 1, -1)/√3, (-1, -1, 1)/√3 (computed in the test).
"""

import numpy as np
import pytest
import scipy.linalg
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site

from bbh_tri import evolve, lattice, measure, operators, states
from bbh_tri.hamiltonian import Model

DENSE = operators.spin1("dense")
DENSE_C = operators.spin1("dense", default_dtype="complex128")
SQRT3 = lattice.geometry("sqrt3")
CELL22 = lattice.geometry("patch", (2, 2))
TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)


def coherent(n):
    """Spin-1 coherent state along unit vector n, canonical basis."""
    Sy, Sz = (operators.as_matrix(DENSE, o) for o in (DENSE.sy(), DENSE.sz()))
    th, ph = np.arccos(n[2]), np.arctan2(n[1], n[0])
    return scipy.linalg.expm(-1j * ph * Sz) @ scipy.linalg.expm(-1j * th * Sy) @ np.array([1, 0, 0])


def tet_index(s):
    return 2 * (s[0] % 2) + s[1] % 2


@pytest.fixture(scope="module")
def tetrahedral():
    psi = fpeps.product_peps(CELL22, {s: operators.state_vector(DENSE_C, coherent(TET[tet_index(s)]))
                                      for s in CELL22.sites()})
    return psi, measure.ctm_env(psi, 4)


@pytest.fixture(scope="module")
def entangled():
    """Dense D = 3 sqrt3 state, BP-evolved at theta = 0.2 pi from the 120° seed (m(K) ≈ 0.71)."""
    psi = states.product_peps(SQRT3, states.seed_vectors("AFM3_120"), DENSE)
    evolve.evolve(psi, Model(0.2 * np.pi), DENSE, 3, schedule=(0.1,), max_steps=40)
    env, info = measure.ctm_env(psi, 12, corner_tol=1e-11, return_info=True)  # chi = 18 does not converge
    assert info.converged
    return psi, env


def test_commensurate_momenta():
    cells = [(SQRT3, {"Gamma", "K"}), (CELL22, {"Gamma", "M1", "M2", "M3"}),
             (lattice.geometry("patch", (6, 6)), set(measure.MOMENTA))]
    for geo, ok in cells:
        assert {k for k, q in measure.MOMENTA.items() if measure.commensurate(geo, q)} == ok


def test_incommensurate_momentum_raises():
    psi = states.product_peps(SQRT3, states.seed_vectors("FM"), DENSE)
    with pytest.raises(ValueError, match="incommensurate"):
        measure.ordered_moments(psi, measure.ctm_env(psi, 2), DENSE, momenta=["M1"])


@pytest.mark.parametrize("seed, expected", [
    ("FM", {"Gamma": (1, 1 / np.sqrt(3)), "K": (0, 0)}),
    ("AFM3_120", {"Gamma": (0, None), "K": (1, None)}),
    ("FQ", {"Gamma": (0, 2 / np.sqrt(3)), "K": (0, 0)}),
    ("AFQ", {"Gamma": (0, 0), "K": (0, 2 / np.sqrt(3))}),
    ("SU3", {"Gamma": (0, 0), "K": (np.sqrt(2 / 3), np.sqrt(2 / 3))}),
])
def test_ordered_moments_of_product_states(seed, expected):
    psi = states.product_peps(SQRT3, states.seed_vectors(seed), DENSE)
    got = measure.ordered_moments(psi, measure.ctm_env(psi, 2), DENSE)
    for q, (m, Q) in expected.items():
        assert got[q][0] == pytest.approx(m, abs=1e-12)
        if Q is not None:
            assert got[q][1] == pytest.approx(Q, abs=1e-12)


@pytest.mark.parametrize("geo", [SQRT3, CELL22], ids=["sqrt3", "2x2"])
def test_ordered_moments_add_up_over_cell_momenta(geo):
    """Parseval: Γ, K (sqrt3) and Γ, M1-3 (2x2) are complete, so sum_q X(q)^2 = mean |<X>|^2."""
    psi = states.random_product_peps(geo, DENSE, seed=3)
    env = measure.ctm_env(psi, 2)
    om = measure.ordered_moments(psi, env, DENSE)
    local = measure.order_parameters(psi, env, DENSE)
    for i in (0, 1):
        assert sum(v[i] ** 2 for v in om.values()) == pytest.approx(np.mean([v[i] ** 2 for v in local.values()]), abs=1e-12)


def test_tetrahedral_product_state(tetrahedral):
    psi, env = tetrahedral
    om = measure.ordered_moments(psi, env, DENSE_C)
    assert om["Gamma"][0] == pytest.approx(0, abs=1e-12)
    for q in ("M1", "M2", "M3"):
        assert om[q][0] == pytest.approx(1 / np.sqrt(3), abs=1e-12)
    chi = measure.chiralities(psi, env, DENSE_C)
    for name, offs in measure.TRIANGLES.items():
        for s in CELL22.sites():
            n = [TET[tet_index((s[0] + dr, s[1] + dc))] for dr, dc in offs]
            assert chi[name][s] == pytest.approx(np.dot(n[0], np.cross(n[1], n[2])), abs=1e-12)
    assert chi["up"][Site(0, 0)] == pytest.approx(4 / (3 * np.sqrt(3)), abs=1e-12)


@pytest.mark.parametrize("dirn", ["h", "v"])
def test_product_state_correlations_factorize(dirn):
    psi = states.random_product_peps(SQRT3, DENSE, seed=1)
    env = measure.ctm_env(psi, 2)
    conn = measure.correlations(psi, env, DENSE, dirn=dirn, rmax=4, connected=True)
    for kind in conn:
        for v in conn[kind].values():
            assert np.abs(v).max() < 1e-12
    full = measure.correlations(psi, env, DENSE, dirn=dirn, rmax=4)
    assert np.abs(full["SS"][Site(0, 0)]).max() > 1e-2  # the subtraction removed something


def qq_operator(ops):
    S = operators.ss(ops)
    return 2 * operators.two_site_product(S, S) + S


def test_correlations_at_short_distance_match_rdms_and_windows(entangled):
    """r = 1 against the bond rdms (h, v), r = 2 against an exact 1x3 window."""
    psi, env = entangled
    for dirn, cls in (("h", "h"), ("v", "v")):
        c = measure.correlations(psi, env, DENSE, dirn=dirn, rmax=2)
        for b in lattice.bonds(SQRT3, cls):
            rdm = measure.bond_rdm(psi, env, b, cls)
            assert c["SS"][b[0]][0] == pytest.approx(np.real(measure.rdm_expectation(operators.ss(DENSE), rdm)), abs=1e-12)
            qq = np.real(measure.rdm_expectation(qq_operator(DENSE), rdm)) + measure.QQ_SHIFT
            assert c["QQ"][b[0]][0] == pytest.approx(qq, abs=1e-12)
    c = measure.correlations(psi, env, DENSE, dirn="h", rmax=2)
    s0, s2 = Site(0, 0), Site(0, 2)
    win = sum(k * env.measure_nsite_exact(A, B, sites=[s0, s2]) for k, A, B in measure.bbh_terms(DENSE, 1, 0))
    assert abs(c["SS"][s0][1]) > 1e-3
    assert c["SS"][s0][1] == pytest.approx(np.real(win), abs=1e-12)


def test_correlation_length_bounds_the_decay(entangled):
    """Connected correlations decay no slower than the leading transfer-matrix length xi_1 (period
    L = 3 along a1): over r = 3 -> 9, -6 / ln|C(9) / C(3)| <= xi_1, and not far below it."""
    psi, env = entangled
    xi = measure.correlation_lengths(env, "h", k=4)
    assert np.all(np.diff(xi) <= 1e-12) and xi[0] > 0
    c = measure.correlations(psi, env, DENSE, dirn="h", rmax=9, connected=True)
    for kind in ("SS", "QQ"):
        v = c[kind][Site(0, 0)]
        xi_eff = -6 / np.log(abs(v[8] / v[2]))
        assert 0.5 * xi[0] < xi_eff <= xi[0]


@pytest.mark.parametrize("sym, seed, t", [("U1", "FQ", 1.6),
                                          pytest.param("U1xU1", "SU3", 0.25, marks=pytest.mark.slow),
                                          pytest.param("Z2xZ2", "AFQ", 0.35, marks=pytest.mark.slow)])
def test_symmetric_configs_match_dense(sym, seed, t):
    ops = operators.spin1(sym)
    psi = states.product_peps(SQRT3, states.seed_vectors(seed), ops)
    evolve.evolve(psi, Model(np.pi * t), ops, 2, schedule=(0.1,), max_steps=20)
    env = measure.ctm_env(psi, 4)  # converges for all three; U1 at chi = 8 does not
    dense = states.to_dense(psi, ops)
    denv = measure.ctm_env(dense, 4)
    a = measure.correlations(psi, env, ops, rmax=3, connected=True)
    b = measure.correlations(dense, denv, DENSE, rmax=3, connected=True)
    for kind in a:
        for s in a[kind]:
            assert np.allclose(a[kind][s], b[kind][s], atol=1e-11)
    assert abs(a["SS"][Site(0, 0)][0]) > 1e-2
    ca, cb = measure.chiralities(psi, env, ops), measure.chiralities(dense, denv, DENSE)
    assert all(ca[k][s] == pytest.approx(cb[k][s], abs=1e-12) for k in ca for s in ca[k])
    oa, ob = measure.ordered_moments(psi, env, ops), measure.ordered_moments(dense, denv, DENSE)
    assert all(np.allclose(oa[q], ob[q], atol=1e-11) for q in oa)  # the symmetric state has no other components


def test_chirality_of_an_entangled_complex_state(tetrahedral):
    """The S± expansion (measure_2x2) against eps_abc Sa Sb Sc (measure_nsite_exact) on a complex
    D = 2 state with nonzero chirality, NN-evolved from the tetrahedral product state."""
    psi = tetrahedral[0].copy()
    evolve.evolve(psi, Model(0.2 * np.pi), DENSE_C, 2, schedule=(0.1,), max_steps=10)
    env = measure.ctm_env(psi, 8)
    chi = measure.chiralities(psi, env, DENSE_C)
    S = list(operators.spin_vector(DENSE_C).values())
    eps = [((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)]
    for name, offs in measure.TRIANGLES.items():
        for s in CELL22.sites():
            tri = [Site(s[0] + dr, s[1] + dc) for dr, dc in offs]
            ref = sum(sg * env.measure_nsite_exact(S[a], S[b], S[c], sites=tri) for (a, b, c), sg in eps)
            assert abs(chi[name][s]) > 1e-2
            assert chi[name][s] == pytest.approx(np.real(ref), abs=1e-11)
