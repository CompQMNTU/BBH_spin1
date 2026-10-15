"""Tests for bbh_tri.yastn_compat (workarounds for YASTN bugs, docs/yastn_findings.md)."""

import warnings

import numpy as np
import pytest
import yastn
import yastn.tn.fpeps as fpeps

from bbh_tri.yastn_compat import peps_to_dict
from yastn_utils import spin1, svd_directional_derivative


@pytest.fixture(scope="module")
def ops():
    return spin1("dense")


def _compat_svd(spectrum, keep, seed=0):
    """(AD, FD, number of DegenerateSVDWarnings) with the patched SVD backward."""
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    with exact_svd_backward(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DegenerateSVDWarning)
        ad, fd = svd_directional_derivative(spectrum, keep, seed=seed)
    return ad, fd, sum(issubclass(x.category, DegenerateSVDWarning) for x in w)


@pytest.mark.ad
@pytest.mark.parametrize("scale", [1.0, 1e-5, 1e-10])
def test_exact_svd_backward_small_singular_values(scale):
    ad, fd, n_warn = _compat_svd(scale * 0.5 ** np.arange(9), keep=6)
    assert ad == pytest.approx(fd, rel=1e-5)
    assert n_warn == 0


@pytest.mark.ad
@pytest.mark.parametrize("gap", [1e-9, 1e-8])
def test_exact_svd_backward_close_but_resolved_pair(gap):
    ad, fd, n_warn = _compat_svd([1.0, 0.7, 0.7 - gap, 0.3, 0.2, 0.1], keep=4)
    assert ad == pytest.approx(fd, rel=1e-6)
    assert n_warn == 0


@pytest.mark.ad
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("gap", [0.0, 2.0**-52, 1e-14, 1e-12])
def test_exact_svd_backward_warns_on_numerical_degeneracy(gap, seed):
    """A kept pair degenerate to roundoff: the gradient is unreliable (0/0 limit), so it must warn."""
    _, _, n_warn = _compat_svd([1.0, 0.7, 0.7 - gap, 0.3, 0.2, 0.1], keep=4, seed=seed)
    assert n_warn > 0


@pytest.mark.parametrize("kw,n_sites", [({}, 3), ({"full_patch": True}, 9),
                                        ({"dims": (2, 3), "boundary": "obc", "full_patch": True}, 6)])
def test_peps_to_dict_keeps_geometry(ops, kw, n_sites):
    geo = fpeps.TriangularLattice(**kw)
    ops.random_seed(seed=0)
    psi = fpeps.product_peps(geo, ops.vec_z(val=1))
    for s in psi.sites():
        psi[s] = yastn.rand(ops.config, legs=psi[s].get_legs())
    psi2 = fpeps.Peps.from_dict(peps_to_dict(psi))
    g = psi2.geometry
    assert len(geo.sites()) == len(g.sites()) == n_sites
    assert (g.dims, g.boundary, g.full_patch) == (geo.dims, geo.boundary, geo.full_patch)
    assert list(g.sites()) == list(geo.sites())
    assert all(list(g.bonds(d)) == list(geo.bonds(d)) for d in "hvd")
    assert all(np.allclose(psi2[s].to_numpy(), psi[s].to_numpy()) for s in geo.sites())


@pytest.mark.ad
def test_degenerate_pair_with_symmetric_gradient_warns():
    """Codex review of PR #4: A = diag(1, .7, .7, .3), keep 3, L = R[1,2] + R[2,1] of the truncated
    reconstruction R. Both skew numerators vanish, the backward returns 0 (FD: 2); it must warn."""
    import torch
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    cfg = yastn.make_config(backend="torch", default_dtype="float64")

    def loss(A):
        T = yastn.Tensor(config=cfg, s=(1, -1))
        T.set_block(val=A, Ds=(4, 4))
        U, S, V = yastn.linalg.svd_with_truncation(T, axes=(0, 1), D_total=3, policy="fullrank", tol=0)
        R = U.to_dense() @ S.to_dense() @ V.to_dense()
        return R[1, 2] + R[2, 1]

    A = torch.diag(torch.tensor([1.0, 0.7, 0.7, 0.3], dtype=torch.float64)).requires_grad_()
    with exact_svd_backward(), pytest.warns(DegenerateSVDWarning):
        loss(A).backward()


@pytest.mark.ad
def test_degeneracy_threshold_is_block_local():
    """Codex review of PR #4: a U1 block with spectrum (2e-12, 1e-12) next to a norm-1 block is
    well resolved (relative gap 1/2); its gradient is accurate and must not warn."""
    import torch
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    cfg = yastn.make_config(backend="torch", default_dtype="float64", sym="U1")
    g = torch.Generator().manual_seed(0)

    def block(spectrum):
        n = len(spectrum)
        Q1, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        Q2, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        return Q1 @ torch.diag(torch.tensor(spectrum, dtype=torch.float64)) @ Q2.T

    B0, B1 = block([1.0, 0.5, 0.2]), block([2e-12, 1e-12])
    X = torch.randn(2, 2, generator=g, dtype=torch.float64)
    W = torch.randn(2, 2, generator=g, dtype=torch.float64)
    d = torch.randn(2, 2, generator=g, dtype=torch.float64)

    def loss(b1):
        T = yastn.Tensor(config=cfg, s=(1, -1))
        T.set_block(ts=(0, 0), val=B0, Ds=(3, 3))
        T.set_block(ts=(1, 1), val=b1, Ds=(2, 2))
        U, S, V = yastn.linalg.svd(T, axes=(0, 1))
        u1 = U.to_dense()[3:, 3:4]  # leading singular vector of the small block (dense order: charge 0, then 1)
        r1 = (U @ S @ V).to_dense()[3:, 3:]
        return (r1 * X).sum() / 1e-12 + (W @ u1 @ u1.T).trace()

    b1 = B1.clone().requires_grad_()
    with exact_svd_backward(), warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateSVDWarning)
        loss(b1).backward()
    eps = 1e-6 * 1e-12
    with torch.no_grad():
        fd = (loss(B1 + eps * d) - loss(B1 - eps * d)).item() / (2 * eps)
    assert (b1.grad * d).sum().item() == pytest.approx(fd, rel=1e-6)


def _small_pair_loss_backward(small):
    """A = diag(1, .7, small, small), L = R01 + R10 + R23 + R32 of R = U S V; the true dL/dA23 is 1."""
    import torch
    cfg = yastn.make_config(backend="torch", default_dtype="float64")
    T = yastn.Tensor(config=cfg, s=(1, -1))
    A = torch.diag(torch.tensor([1.0, 0.7, small, small], dtype=torch.float64)).requires_grad_()
    T.set_block(val=A, Ds=(4, 4))
    U, S, V = yastn.linalg.svd(T, axes=(0, 1))
    R = (U @ S @ V).to_dense()
    (R[0, 1] + R[1, 0] + R[2, 3] + R[3, 2]).backward()
    return A.grad


@pytest.mark.ad
@pytest.mark.parametrize("small", [1e-8, 1e-22, 1e-300])
def test_degenerate_small_pair_warns_despite_small_raw_weight(small):
    """Codex review of PR #4 (3rd and 4th pass): the backward gives dL/dA23 = 0 instead of 1, while
    the raw gradient into the small degenerate pair is ~small of the total. The 1/sigma-scaled
    weight must flag it at any nonzero scale, including below eps * sigma_max."""
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    with exact_svd_backward(), pytest.warns(DegenerateSVDWarning):
        _small_pair_loss_backward(small)


@pytest.mark.ad
@pytest.mark.xfail(strict=True, reason="undetectable: gU, gV carry a factor sigma = 0, so no information about the "
                   "lost term reaches the backward (docs/yastn_findings.md U4). AD runs must truncate with tol > 0.")
def test_degenerate_zero_pair_is_undetectable():
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    with exact_svd_backward(), pytest.warns(DegenerateSVDWarning):
        _small_pair_loss_backward(0.0)


def _diag_svd_grad(diag, keep, tol, loss_of_R_P):
    """Gradient of loss(R, P) for A = diag(...), R = U S V (truncated), P = U U^T; returns (grad, n_warn)."""
    import torch
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    cfg = yastn.make_config(backend="torch", default_dtype="float64")
    A = torch.diag(torch.tensor(diag, dtype=torch.float64)).requires_grad_()
    with exact_svd_backward(), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DegenerateSVDWarning)
        T = yastn.Tensor(config=cfg, s=(1, -1))
        T.set_block(val=A, Ds=tuple(A.shape))
        U, S, V = yastn.linalg.svd_with_truncation(T, axes=(0, 1), D_total=keep, policy="fullrank", tol=tol)
        u = U.to_dense()
        loss_of_R_P(u @ S.to_dense() @ V.to_dense(), u @ u.T).backward()
    return A.grad, sum(issubclass(x.category, DegenerateSVDWarning) for x in w)


@pytest.mark.ad
def test_degenerate_pair_warns_when_other_terms_dominate():
    """Codex review of PR #4 (5th pass): a projector term on small singular values inflated the
    denominator of the old fraction test, hiding the .7 pair (AD 0 vs true 1). Any gradient into a
    degenerate pair must warn."""
    _, n_warn = _diag_svd_grad([1, .7, .7, 2e-8, 1e-8, 1e-9], 5, 1e-10,
                               lambda R, P: R[1, 2] + R[2, 1] + P[3, 4] + P[4, 3])
    assert n_warn > 0


@pytest.mark.ad
def test_discarded_subnormal_gap_gives_no_nan():
    """Codex review of PR #4 (5th pass): discarded pair (1e-300, 1e-300 - 1e-310) has a gap whose
    reciprocal overflows; inf * 0 gave NaN in the whole gradient. Discarded columns carry no gradient."""
    import torch
    grad, n_warn = _diag_svd_grad([1, 1e-300, 1e-300 - 1e-310], 1, 0, lambda R, P: R[0, 0])
    assert not torch.isnan(grad).any()
    assert grad[0, 0].item() == pytest.approx(1.0)
    assert n_warn == 0
