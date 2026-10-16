"""M3 (SPEC §6): NN imaginary-time evolution on the sqrt3 x sqrt3 cell, dense."""

import numpy as np
import pytest

from bbh_tri import evolve, lattice, measure, operators, states
from bbh_tri.hamiltonian import Model


def run(theta, D, env="BP", seed=0, **kw):
    ops = operators.spin1("dense")
    psi = states.random_product_peps(lattice.geometry("sqrt3"), ops, seed=seed)
    return psi, ops, evolve.evolve(psi, Model(theta), ops, D, env=env, **kw)


@pytest.mark.parametrize("theta", [np.pi, pytest.param(0.6 * np.pi, marks=pytest.mark.slow),
                                   pytest.param(1.2 * np.pi, marks=pytest.mark.slow)])
def test_fm_region_energy(theta):
    """SPEC §6 M3 (a): in the FM phase (pi/2 < theta < 5pi/4) E/site = 3(cos + sin) from a random start.

    Only the energy is checked: the whole maximal-spin multiplet is degenerate, and from a random
    start D = 2 converges to a superposition of FM directions with reduced local moment (m = 0.74
    at theta = pi), at exactly the FM energy.
    """
    _, _, r = run(theta, D=2, seed=1, max_steps=200)
    assert r.energy == pytest.approx(3 * (np.cos(theta) + np.sin(theta)), abs=1e-6)


def test_fm_seed_keeps_full_moment():
    """Started from the FM product seed, the state stays the m = 1 product state."""
    ops = operators.spin1("dense")
    geo = lattice.geometry("sqrt3")
    psi = states.product_peps(geo, states.seed_vectors("FM"), ops)
    r = evolve.evolve(psi, Model(np.pi), ops, 2, max_steps=40)
    assert r.energy == pytest.approx(-3, abs=1e-10)
    env = measure.ctm_env(psi, chi=4)
    assert all(m == pytest.approx(1, abs=1e-10) for m, _ in measure.order_parameters(psi, env, ops).values())


@pytest.mark.slow
def test_heisenberg_d2_below_product_state():
    """SPEC §6 M3 (b), D = 2: below the D = 1 value -3/2 (120° product state, SPEC §3), above the
    D -> inf value -1.8368(5) (Niesen & Corboz, PRB 97, 245146), and C3-symmetric within 5%."""
    _, _, r = run(0.0, D=2, max_steps=200)
    assert -1.8368 < r.energy < -1.5 - 0.2
    e = list(r.class_energies.values())
    assert max(e) - min(e) < 0.05 * abs(np.mean(e))
    assert all(s.converged for s in r.stages)


def test_tensors_stay_normalized():
    """theta = 1.5 pi grows the unnormalized tensors by ~1e5 per 20 steps (NaN after ~1200 steps)."""
    psi, _, r = run(1.5 * np.pi, D=2, max_steps=20, schedule=(0.05,))
    assert np.isfinite(r.energy)
    assert all(psi[s].norm() == pytest.approx(1, abs=1e-12) for s in psi.sites())
