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


def _compat_svd(spectrum, keep, seed=0, kept_degenerate="warn"):
    """(AD, FD, number of DegenerateSVDWarnings) with the patched SVD backward."""
    from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward
    with exact_svd_backward(kept_degenerate), warnings.catch_warnings(record=True) as w:
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
@pytest.mark.parametrize("tail", [[1e-300, 1e-300 - 1e-310], [1.2e-308, 6e-309]])
def test_discarded_subnormal_gap_gives_no_nan(tail):
    """Codex review of PR #4 (5th, 6th pass): a discarded pair whose 1/gap overflows (1e-300 tail),
    or whose finite F, G overflow in F + G (1e-308 tail), gave inf * 0 = NaN in the whole gradient.
    Discarded columns carry no gradient and must contribute exactly 0."""
    import torch
    grad, n_warn = _diag_svd_grad([1, *tail], 1, 0, lambda R, P: R[0, 0])
    assert not torch.isnan(grad).any()
    assert grad[0, 0].item() == pytest.approx(1.0)
    assert n_warn == 0


@pytest.mark.ad
def test_active_pair_with_overflowing_gap_warns():
    """Codex review of PR #4 (6th pass): diag(1e-307, 9.9e-308) is well resolved (relative gap 1e-2)
    but 1/gap overflows and would be dropped silently (AD 0 vs true 1 for L = R01 + R10)."""
    _, n_warn = _diag_svd_grad([1e-307, 9.9e-308], 2, 0, lambda R, P: R[0, 1] + R[1, 0])
    assert n_warn > 0


@pytest.mark.ad
@pytest.mark.parametrize("seed", range(2))
def test_zero_mode_drops_small_kept_degenerate_pair(seed):
    """kept_degenerate='zero': a kept pair at 1e-4 sigma_max is zeroed; the dropped 0/0 term costs
    ~1e-5 relative on this loss (it depends on how the pair splits)."""
    ad, fd, n_warn = _compat_svd([1, .5, .3, 1e-4, 1e-4, 1e-5], keep=5, seed=seed, kept_degenerate="zero")
    assert n_warn == 0
    assert ad == pytest.approx(fd, rel=1e-4)


@pytest.mark.ad
@pytest.mark.parametrize("spectrum,keep", [([1, .7, .7, .3, .2, .1], 4),     # large kept pair
                                           ([1, .5, 1e-4, 1e-4, 1e-5], 3)])  # small pair straddling the cut
def test_zero_mode_still_warns_for_large_or_straddling_pairs(spectrum, keep):
    _, _, n_warn = _compat_svd(spectrum, keep=keep, kept_degenerate="zero")
    assert n_warn > 0


@pytest.mark.ad
def test_zero_mode_error_is_not_bounded_by_pair_size():
    """Codex counterexample (PR #8): the dropped term is the loss's sensitivity to mixing within the
    pair, O(1) here although the pair is at 1e-4. L = R[1,2] + R[2,1] of the reconstruction R = U S Vh
    of diag(1, 1e-4, 1e-4) has derivative 2 along dA[1,2] = dA[2,1] = 1; the zero mode gives 0
    (silently, which is why Chain FD-checks gradients after zeroing) and counts the pair."""
    import torch
    import yastn
    from bbh_tri.yastn_compat import exact_svd_backward, zeroed_pairs
    cfg = yastn.make_config(backend="torch", default_dtype="float64")
    A = torch.diag(torch.tensor([1.0, 1e-4, 1e-4], dtype=torch.float64)).requires_grad_()
    T = yastn.Tensor(config=cfg, s=(1, -1))
    T.set_block(val=A, Ds=(3, 3))
    z0 = zeroed_pairs()
    with exact_svd_backward("zero"), warnings.catch_warnings():
        warnings.simplefilter("error")
        U, S, V = yastn.linalg.svd(T, axes=(0, 1))
        R = (U @ S @ V).to_dense()
        (R[1, 2] + R[2, 1]).backward()
    assert (A.grad[1, 2] + A.grad[2, 1]).item() == pytest.approx(0.0, abs=1e-12)  # exact: 2
    assert zeroed_pairs() - z0 == 1


def _diag(blocks, sym="dense"):
    """Diagonal yastn tensor (numpy) with the given values per charge block."""
    import yastn
    cfg = yastn.make_config(sym=sym)
    S = yastn.Tensor(config=cfg, s=(1, -1), isdiag=True)
    for t, v in blocks.items():
        S.set_block(ts=t, Ds=len(v), val=np.asarray(v, dtype=float))
    return S


def test_gap_safe_truncation_never_cuts_a_near_degenerate_pair():
    from bbh_tri.yastn_compat import gap_safe_truncation
    S = _diag({(): [1.0, 0.3, 1e-3, 1e-8 + 5e-11, 1e-8, 1e-9]})
    import yastn
    assert int(np.sum(yastn.linalg.truncation_mask(S, tol=0, D_total=4).to_numpy())) == 4  # plain cut splits the pair
    assert int(np.sum(gap_safe_truncation(0.0, 4)(S).to_numpy())) == 3  # drops the whole pair
    assert int(np.sum(gap_safe_truncation(0.0, 5)(S).to_numpy())) == 5  # cut below the pair: unchanged


def test_gap_safe_truncation_keeps_multiplets_across_blocks():
    """A doublet split over two U1 sectors at the cut: plain truncation keeps one member, the
    multiplet-aware mask keeps or drops both."""
    import yastn
    from bbh_tri.yastn_compat import gap_safe_truncation
    S = _diag({(0,): [1.0, 0.2], (1,): [0.5, 0.2 * (1 + 1e-9)], (-1,): [0.05]}, sym="U1")
    plain = yastn.linalg.truncation_mask(S, tol=0, D_total=3)
    safe = gap_safe_truncation(0.0, 3, eps_multiplet=1e-6)(S)
    count = lambda m: sum(int(np.sum(m[t])) for t in m.struct.t)
    assert count(plain) == 3 and count(safe) == 2


def test_gap_safe_truncation_block_backoff_keeps_cross_block_multiplet():
    """Codex counterexample (PR #10): blocks A = [1, 2e-8, 1.995e-8], B = [2e-8], tol 1e-8, chi 3.
    A per-block backoff after the global multiplet cut dropped A's 2e-8 but kept B's, splitting the
    exact doublet. With both conditions checked at one global cut, the doublet goes together."""
    from bbh_tri.yastn_compat import gap_safe_truncation
    S = _diag({(0,): [1.0, 2e-8, 1.995e-8], (1,): [2e-8]}, sym="U1")
    mask = gap_safe_truncation(1e-8, 3, eps_multiplet=1e-6)(S)
    kept = {t[0]: sorted(np.asarray(S[t]).ravel()[np.asarray(mask[t]).ravel()].tolist()) for t in S.struct.t}
    assert kept[1] == [x for x in kept[0] if x == 2e-8]  # the doublet is kept or dropped whole
    assert kept == {0: [1.0], 1: []}  # here dropped: keeping it would split A's 2e-8 / 1.995e-8 pair


def _cycle_state():
    """D = 4 state at theta = pi/4 (dense copy of a U1xU1 AD line-search trial point, M4) on which
    CTM with a plain chi = 32 cut cycles between two truncations of a multiplet."""
    import os
    import yastn
    import yastn.tn.fpeps as fpeps
    from bbh_tri import lattice, operators
    data = np.load(os.path.join(os.path.dirname(__file__), "data", "d4_pi4_ctm_cycle.npz"))
    cfg = operators.spin1("dense").config
    geo = lattice.geometry("sqrt3")
    tensors = {}
    for k, s in enumerate(geo.sites()):
        a = data[f"s{k}"]
        T = yastn.Tensor(config=cfg, s=(-1, 1, 1, -1, 1, -1))
        T.set_block(val=a, Ds=a.shape)
        tensors[s] = T.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
    return fpeps.Peps(geo, tensors=tensors)


@pytest.mark.slow
def test_multiplet_safe_cut_stops_ctm_cycle():
    from bbh_tri import measure
    from bbh_tri.yastn_compat import gap_safe_truncation
    psi = _cycle_state()
    _, plain = measure.ctm_env(psi, 32, corner_tol=1e-9, max_sweeps=60, return_info=True)
    _, safe = measure.ctm_env(psi, 32, corner_tol=1e-9, max_sweeps=60, return_info=True,
                              truncation_f=gap_safe_truncation(0.0, 32, eps_multiplet=1e-6))
    assert not plain.converged  # two-cycle: corner spectra change by ~2.5e-2 and ~1e-4 on alternate sweeps
    assert safe.converged
