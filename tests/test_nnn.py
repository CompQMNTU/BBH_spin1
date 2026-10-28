"""M5: NNN (J2, K2) evolution and measurement (SPEC §6 M5).

Acceptance (a), gate routing against exact vectors: tests/test_gates.py::test_gates_on_obc_patch_match_exact
(every NNN class and path variant) and tests/test_routing.py (single path gates). Here:
- measured energies of an entangled NNN-evolved state on OBC patches equal the exact expectation value;
- (b) D = 1 energies with J2, K2 on the patch cells match the product-state minimizer
  (tests/test_product_states.py covers the sqrt3 cell);
- (c) the converged energy does not depend on the gate path (within truncation error);
- evolution runs on the (2, 2) and (6, 6) cells (SPEC §4.4).
NNN gates cannot be routed on the sqrt3 cell (every NNN pair shares a tensor; lattice.gate_paths).
"""

import numpy as np
import pytest

from bbh_tri import evolve, gates, lattice, measure, operators, states
from bbh_tri.hamiltonian import Model
from yastn_utils import apply_2site, peps_vector

MODEL = Model(0.7, J2=0.4, K2=-0.3, J_anis=0.6)


def exact_energy(psi, dims, model, ops):
    """<H>/<psi|psi> per site from the dense state vector of a finite PEPS, bonds from lattice.bonds."""
    p = operators.basis_permutation(ops)
    v, order = peps_vector(psi, *dims)
    v = v[np.ix_(*[p] * v.ndim)]
    Hv = np.zeros_like(v)
    for cls, h in model.bond_hamiltonians(ops).items():
        H = operators.as_matrix(ops, h)
        for a, b in lattice.bonds(psi.geometry, cls):
            Hv = Hv + apply_2site(v, order, H, a, b)
    return float(np.vdot(v, Hv).real / np.vdot(v, v).real) / len(order)


@pytest.mark.parametrize("dims", [(2, 3), (3, 2)])  # 2x3 holds 'nnn_k1', 3x2 'nnn_k2'
def test_energy_on_obc_patch_matches_exact_vector(dims):
    """After a few truncated Trotter steps with NNN gates (an entangled D = 3 state), the CTM
    energy per site, all classes including the knight windows, equals the exact one."""
    ops = operators.spin1("dense")
    geo = lattice.geometry("obc", dims)
    psi = states.random_product_peps(geo, ops, seed=4)
    from yastn.tn.fpeps import EnvBP, evolution_step_
    env = EnvBP(psi)
    for v in range(3):
        evolution_step_(env, gates.trotter2(gates.model_gates(geo, MODEL, ops, 0.1, v)), opts_svd={"D_total": 3, "tol": 1e-12})
        evolve.normalize_(psi)
    assert max(d for s in psi.sites() for d in psi[s].get_shape()[:4]) == 3
    e, by_class = measure.energy(psi, measure.ctm_env(psi, chi=81), ops, MODEL)
    assert {c for c in lattice.NNN_CLASSES if c in by_class} == {c for c in lattice.NNN_CLASSES if lattice.bonds(geo, c)}
    assert e == pytest.approx(exact_energy(psi, dims, MODEL, ops), abs=1e-10)


@pytest.mark.parametrize("dims", [(3, 3), (6, 6)])
def test_product_state_on_patch_cells_matches_mean_field(dims):
    """(b) The optimized 3-sublattice product state on a patch cell: CTM energy = minimizer's."""
    ops = operators.spin1("dense", default_dtype="complex128")
    model = Model(0.3, J2=0.4, K2=-0.2)
    e_mf, vs = states.optimize_product_state(model, restarts=2)
    psi = states.product_peps(lattice.geometry("patch", dims), vs, ops)
    e, by_class = measure.energy(psi, measure.ctm_env(psi, chi=1), ops, model)
    assert set(by_class) == set(model.couplings())
    assert e == pytest.approx(e_mf, abs=1e-10)


@pytest.mark.parametrize("kind", ["FM", "FQ"])
def test_uniform_product_state_on_2x2_cell(kind):
    """(b) on the (2, 2) cell, which cannot hold 3-sublattice order: uniform seeds."""
    ops = operators.spin1("dense")
    model = Model(1.2, J2=0.3, K2=0.1)
    psi = states.product_peps(lattice.geometry("patch", (2, 2)), states.seed_vectors(kind), ops)
    e, _ = measure.energy(psi, measure.ctm_env(psi, chi=1), ops, model)
    assert e == pytest.approx(states.product_energy(model, states.seed_vectors(kind)), abs=1e-12)


@pytest.mark.slow
def test_energy_invariant_under_gate_path():
    """(c) theta = 0, J2 = 0.1, D = 2, (3, 3) cell: each fixed NNN path and alternating paths
    converge to the same energy within the truncation error (here ~5e-2 per step; measured
    spread ~4e-4; ~4 min)."""
    ops = operators.spin1("dense")
    model = Model(0.0, J2=0.1)
    out = {}
    for pv in (0, 1, None):
        psi = states.random_product_peps(lattice.geometry("patch", (3, 3)), ops, seed=1)
        r = evolve.evolve(psi, model, ops, 2, schedule=(0.1,), max_steps=240, tol=1e-6, path_variant=pv)
        assert r.stages[-1].converged
        out[pv] = r.energy
    assert max(out.values()) - min(out.values()) < 1e-3, out
    assert all(e < -1.5 for e in out.values())  # well below the D = 1 value -1.2 (3 x J2 above -1.5)


@pytest.mark.slow
@pytest.mark.parametrize("dims", [(2, 2), (6, 6)])
def test_evolution_runs_on_patch_cells(dims):
    """SPEC §4.4 cells: a few NNN Trotter steps lower the energy of a random product state."""
    ops = operators.spin1("dense")
    model = Model(0.0, J2=0.1, K2=0.05)
    psi = states.random_product_peps(lattice.geometry("patch", dims), ops, seed=2)
    e0, _ = measure.energy(psi, measure.ctm_env(psi, chi=1), ops, model)
    r = evolve.evolve(psi, model, ops, 2, schedule=(0.1,), max_steps=6, check_every=6)
    assert np.isfinite(r.energy) and r.energy < e0 - 0.1
    assert set(r.class_energies) == set(model.couplings())
