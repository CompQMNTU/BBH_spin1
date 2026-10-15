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


def svd_directional_derivative(scale):
    """(AD, FD) for a gauge-invariant loss of a truncated SVD of a 12x9 matrix with spectrum scale*0.5**k."""
    torch = pytest.importorskip("torch")
    cfg = yastn.make_config(backend="torch", default_dtype="float64")
    g = torch.Generator().manual_seed(0)
    m, n, k = 12, 9, 6
    Ur, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    Vr, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    A0 = Ur[:, :n] @ torch.diag(scale * 0.5 ** torch.arange(n, dtype=torch.float64)) @ Vr.T
    W = torch.randn(m, m, generator=g, dtype=torch.float64)
    dA = torch.randn(m, n, generator=g, dtype=torch.float64)

    def loss(A):
        T = yastn.Tensor(config=cfg, s=(1, -1))
        T.set_block(val=A, Ds=(m, n))
        U, S, V = yastn.linalg.svd_with_truncation(T, axes=(0, 1), D_total=k, policy="fullrank", tol=0)
        u, s, v = U.to_dense(), S.to_dense(), V.to_dense()
        return (W @ u @ u.T).trace() + (u @ s @ v).sum() / scale

    A = A0.clone().requires_grad_()
    loss(A).backward()
    eps = 1e-6 * scale
    with torch.no_grad():
        fd = (loss(A0 + eps * dA) - loss(A0 - eps * dA)).item() / (2 * eps)
    return (A.grad * dA).sum().item(), fd


@pytest.mark.ad
def test_svd_backward_absolute_regularizer_is_wrong_for_small_singular_values():
    ad, fd = svd_directional_derivative(1.0)
    assert ad == pytest.approx(fd, rel=1e-6)  # fine at O(1) scale
    ad, fd = svd_directional_derivative(1e-5)
    assert abs(ad - fd) / abs(fd) > 0.5  # wrong once singular values drop below ~1e-6


@pytest.mark.ad
def test_exact_svd_backward_fixes_small_singular_values():
    from bbh_tri.yastn_compat import exact_svd_backward
    with exact_svd_backward():
        ad, fd = svd_directional_derivative(1e-5)
    assert ad == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("kw", [{"full_patch": True}, {"dims": (2, 3), "boundary": "obc"}])
def test_triangular_to_dict_loses_geometry(ops, kw):
    psi = fpeps.product_peps(fpeps.TriangularLattice(**kw), ops.vec_z(val=1))
    g = fpeps.Peps.from_dict(psi.to_dict()).geometry
    assert (g.dims, g.boundary, g.full_patch) == ((3, 3), "infinite", False)  # constructor defaults


@pytest.mark.parametrize("kw", [{}, {"full_patch": True}, {"dims": (2, 3), "boundary": "obc"}])
def test_peps_to_dict_keeps_geometry(ops, kw):
    import numpy as np
    from bbh_tri.yastn_compat import peps_to_dict
    geo = fpeps.TriangularLattice(**kw)
    ops.random_seed(seed=0)
    psi = fpeps.product_peps(geo, ops.vec_z(val=1))
    for s in psi.sites():
        psi[s] = yastn.rand(ops.config, legs=psi[s].get_legs())
    psi2 = fpeps.Peps.from_dict(peps_to_dict(psi))
    g = psi2.geometry
    assert (g.dims, g.boundary, g.full_patch) == (geo.dims, geo.boundary, geo.full_patch)
    assert list(g.sites()) == list(geo.sites())
    assert all(list(g.bonds(d)) == list(geo.bonds(d)) for d in "hvd")
    assert all(np.allclose(psi2[s].to_numpy(), psi[s].to_numpy()) for s in geo.sites())


def test_full_patch_bonds_without_direction_raises():
    with pytest.raises(TypeError):
        fpeps.TriangularLattice(full_patch=True).bonds()
