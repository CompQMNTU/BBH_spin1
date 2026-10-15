"""Canaries for YASTN behavior documented in docs/yastn_findings.md.

These pin *current* upstream behavior, including bugs we work around. If one fails after a
yastn bump, the workaround may be obsolete or a new problem appeared: update
docs/yastn_findings.md and SPEC §1, then the test.
"""

import importlib.metadata

import pytest
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site
from yastn.tn.fpeps.envs.rdm import measure_rdm_diag

from yastn_utils import converged_ctm, spin1, ss_yastn

PINNED = "1.6.2.dev320+g7ed8ad0e6"


@pytest.fixture(scope="module")
def ops():
    return spin1("dense")


def test_yastn_version_is_pinned():
    assert importlib.metadata.version("yastn") == PINNED


def test_dense_basis_order(ops):
    import numpy as np
    assert np.allclose(np.diag(ops.sz().to_numpy()), [1, 0, -1])


def test_bare_diagonal_gate_raises(ops):
    geo = fpeps.TriangularLattice()
    psi = fpeps.product_peps(geo, ops.vec_z(val=1))
    gate = fpeps.gates.gate_nn_exp(0.1, ops.I(), ss_yastn(ops))
    with pytest.raises(yastn.YastnError, match="not nearest-neighbor"):
        fpeps.evolution_step_(fpeps.EnvBP(psi), [gate._replace(sites=geo.bonds("d")[0])],
                              opts_svd={"D_total": 2})


def test_distribute_emits_bare_diagonal_gates(ops):
    geo = fpeps.TriangularLattice()
    gate = fpeps.gates.gate_nn_exp(0.1, ops.I(), ss_yastn(ops))
    sites = [g.sites for g in fpeps.gates.distribute(geo, gates_nn=gate)]
    assert any(tuple(b) in sites for b in geo.bonds("d"))


def test_measure_nn_rejects_diagonal(ops):
    geo = fpeps.TriangularLattice()
    psi = fpeps.product_peps(geo, ops.vec_z(val=1))
    env = converged_ctm(psi, chi=1)
    with pytest.raises(yastn.YastnError, match="not nearest-neighbor"):
        env.measure_nn(ops.sz(), ops.sz(), bond=geo.bonds("d")[0])


def test_measure_rdm_diag_crashes_on_dense(ops):
    geo = fpeps.TriangularLattice()
    psi = fpeps.product_peps(geo, ops.vec_z(val=1))
    env = converged_ctm(psi, chi=1)
    with pytest.raises(IndexError):
        measure_rdm_diag(Site(0, 0), "anti_diag", psi, env, (ops.sz(), ops.sz()))


def test_obc_triangular_diagonal_bonds_contain_none():
    geo = fpeps.TriangularLattice(dims=(3, 3), boundary="obc", full_patch=True)
    assert any(None in tuple(b) for b in geo.bonds("d"))


def test_no_measure_nsite_exact_oe():
    assert not hasattr(fpeps.EnvCTM, "measure_nsite_exact_oe")
