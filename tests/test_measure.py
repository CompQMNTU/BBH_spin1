"""U2 (SPEC §1): measuring the triangular diagonal bond, plus a D=1 pipeline check.

Reference: SU(3) point theta = pi/4, product state |+1>,|0>,|-1> on the three sublattices has
E/site = 3/sqrt(2) and every bond = 1/sqrt(2) (SPEC §3; h = (P_swap + 1)/sqrt(2)).
"""

import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site
from yastn.tn.fpeps.envs.rdm import rdm2x2_anti_diagonal

from yastn_utils import bbh_bond_yastn, converged_ctm, diag_path, energy_per_class, spin1

THETA_SU3 = np.pi / 4


@pytest.fixture(scope="module")
def ops():
    return spin1("dense")


@pytest.fixture(scope="module")
def geometry():
    return fpeps.TriangularLattice()  # sqrt3 x sqrt3, infinite


@pytest.fixture(scope="module")
def entangled(ops, geometry):
    """A D=2 sqrt3 x sqrt3 state with nontrivial diagonal correlations and its CTM env."""
    psi = fpeps.product_peps(geometry, {s: ops.vec_z(val=1) + 0.3 * ops.vec_z(val=0) + 0.2 * ops.vec_z(val=-1)
                                        for s in geometry.sites()})
    ops.random_seed(seed=0)
    for s in psi.sites():
        psi[s] = psi[s] + 0.2 * yastn.rand(ops.config, legs=psi[s].get_legs())
    gate = fpeps.gates.gate_nn_exp(0.3, ops.I(), bbh_bond_yastn(ops, THETA_SU3))
    gates = [gate._replace(sites=b) for b in list(geometry.bonds("h")) + list(geometry.bonds("v"))]
    gates += [gate._replace(sites=diag_path(b)) for b in geometry.bonds("d")]
    fpeps.evolution_step_(fpeps.EnvNTU(psi), gates, opts_svd={"D_total": 2})
    return psi, converged_ctm(psi, chi=8, corner_tol=1e-12)


def test_diagonal_measurement_routes_agree(ops, geometry, entangled):
    psi, env = entangled
    Sz = ops.sz()
    b = geometry.bonds("d")[0]  # ((x+1,y),(x,y+1))
    via_2x2 = env.measure_2x2(Sz, Sz, sites=b)
    via_exact = env.measure_nsite_exact(Sz, Sz, sites=b)
    rdm = rdm2x2_anti_diagonal(Site(b[1][0], b[0][1]), psi, env)[0].to_numpy()
    via_rdm = np.einsum("ij,jilk,kl->", Sz.to_numpy(), rdm, Sz.to_numpy()) / np.einsum("iikk->", rdm)
    assert abs(via_2x2) > 1e-3  # nontrivial correlation, so agreement means something
    assert via_exact == pytest.approx(via_2x2, abs=1e-12)
    assert via_rdm == pytest.approx(via_2x2, abs=1e-12)


def test_knight_window_runs(ops, entangled):
    """U3 smoke test: a 2x3 window for the (-1,2) NNN bond can be measured (cost at target D: open)."""
    psi, env = entangled
    val = env.measure_nsite_exact(ops.sz(), ops.sz(), sites=[Site(1, 0), Site(0, 2)])
    assert np.isfinite(val)


def test_su3_product_state_energy(ops, geometry):
    """D=1 through the full YASTN pipeline (PEPS -> CTM -> rdm) at theta = pi/4."""
    psi = fpeps.product_peps(geometry, {Site(0, 0): ops.vec_z(val=1),
                                        Site(0, 1): ops.vec_z(val=0),
                                        Site(0, 2): ops.vec_z(val=-1)})
    env = converged_ctm(psi, chi=1)
    e_site, e_class = energy_per_class(psi, env, ops, THETA_SU3)
    assert e_site == pytest.approx(3 / np.sqrt(2), abs=1e-12)
    for k in ("h", "v", "d"):
        assert e_class[k] == pytest.approx(1 / np.sqrt(2), abs=1e-12)
