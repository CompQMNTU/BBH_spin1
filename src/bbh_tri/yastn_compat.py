"""Workarounds for YASTN bugs at the pinned commit (docs/yastn_findings.md).

- SVD backward: ``yastn.backend.linalg.torch_svd_gesdd.safe_inverse`` ignores its ``eps_abs``
  argument and regularizes 1/x as x/(x^2 + 1e-12) with an *absolute* 1e-12. CTM matrices have
  singular values and gaps far below 1e-6, so AD gradients through ctmrg/fp_ctmrg come out wrong
  (2% at chi=8, 40% at chi=16 for a D=2 test state). ``exact_svd_backward`` installs
  :func:`svd_backward`, which uses exact inverses (0 only at exactly 0) and warns with
  :class:`DegenerateSVDWarning` when numerically degenerate singular values carry gradient.
- Serialization: ``TriangularLattice.to_dict`` keeps only ``pattern``, so ``Peps.from_dict``
  rebuilds the default geometry (a full_patch or OBC state silently changes shape).
  ``peps_to_dict`` stores the constructor arguments instead.
"""

import contextlib
import warnings

import numpy as np

import yastn.tn.fpeps as fpeps

# Pairs with |sigma_i - sigma_j| <= DEGENERACY_RTOL * sigma_max(block) are numerically degenerate.
# For a gauge-invariant loss their 1/(sigma_i - sigma_j) term is 0/0 with a finite, nonzero limit
# that roundoff destroys; neither zeroing nor broadening recovers it (kept pair at gap 0..1e-14:
# 1e-4 to 80% gradient error; at 1e-12: up to 6e-4; at >= 1e-9: <= 1e-7). The same holds for 1/sigma
# when sigma = 0 or 1/sigma overflows. svd_backward warns, conservatively, whenever ANY gradient
# reaches a singular vector of such a pair or value. Fraction-based tests were defeated by
# counterexamples (Codex reviews of PR #4), so there is no tolerance: discarded columns receive
# exactly zero gradient and never trigger it (CTM tails at chi = 4-16: no warning). False positives:
# losses that depend on a degenerate pair only through its span (e.g. a projector).
# Exactly zero kept singular values are undetectable (gU, gV vanish with sigma): AD runs must truncate
# with opts_svd tol > 0.
DEGENERACY_RTOL = 1e-10


# How svd_backward treats a near-degenerate pair whose two singular vectors BOTH receive gradient
# (both kept): "warn" (default) or "zero". "zero" drops the pair's 1/(sigma_i - sigma_j) term, whose
# true value is a 0/0 limit (it vanishes only if the loss does not depend on how the pair splits).
# The error is NOT bounded by the pair's size: it is the loss's sensitivity to mixing within the
# pair, which can be O(1) however small the pair is (e.g. L = R[1,2] + R[2,1] of the reconstruction
# R = U S Vh of diag(1, 1e-4, 1e-4): exact derivative 2 along dA[1,2] = dA[2,1] = 1, zero mode 0;
# test_yastn_compat.py). Measured: generic loss, rel. error 1e-2 for a pair at 0.7 sigma_max, ~5e-6 at
# 1e-4; CTM energy at D = 3 with a pair at 2e-4 sigma_max: 2e-8 (test_variational.py). It is
# therefore an empirical approximation: only pairs at sigma <= ZERO_MAX_SIGMA * sigma_max of the block
# are zeroed (larger kept pairs, and pairs straddling the cut, still warn), every zeroed pair is
# counted (:func:`zeroed_pairs`), and ``variational.Chain`` checks its gradient against finite
# differences whenever pairs were zeroed.
_KEPT_DEGENERATE = "warn"
ZERO_MAX_SIGMA = 1e-3
_ZEROED_PAIRS = 0


def zeroed_pairs():
    """Number of degenerate pairs zeroed by ``kept_degenerate='zero'`` since import (monotonic)."""
    return _ZEROED_PAIRS


class DegenerateSVDWarning(RuntimeWarning):
    """Near-degenerate singular values contribute to an SVD gradient, which is then unreliable."""


def _unsafe(x):
    """Entries whose reciprocal is undefined or overflows (x = 0 or |x| < 1/finfo.max)."""
    import torch
    return x.abs() * torch.finfo(x.dtype).max < 1


def _inverse(x, zero):
    """1/x, with 0 where ``zero`` or where 1/x would overflow (avoids inf * 0 = nan downstream)."""
    import torch
    zero = zero | _unsafe(x)
    return torch.where(zero, torch.zeros_like(x), 1.0 / torch.where(zero, torch.ones_like(x), x))


def svd_backward(ctx, gu, gsigma, gvh):
    """Drop-in for ``SVDGESDD.backward`` (same structure, regularization replaced).

    ``ctx.saved_tensors`` = (u, sigma, vh, eps, sigma_max_all_blocks) as set by yastn's
    ``kernel_svd.backward``; both ``eps`` (yastn's ad_decomp_reg) and the all-block maximum are
    ignored: degeneracy is judged against this block's own largest singular value, since each
    block's SVD roundoff is relative to that block.
    """
    import torch
    u, sigma, vh = ctx.saved_tensors[:3]
    m, n, k = u.size(0), vh.size(1), sigma.size(0)
    if u.size(-2) != u.size(-1) or vh.size(-2) != vh.size(-1):
        u = u.narrow(-1, 0, k)
        vh = vh.narrow(-2, 0, k)
        if gu is not None:
            gu = gu.narrow(-1, 0, k)
        if gvh is not None:
            gvh = gvh.narrow(-2, 0, k)

    if gsigma is not None:
        sigma_term = u * gsigma.unsqueeze(-2) @ vh
    else:
        sigma_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)
    if gu is None and gvh is None:
        return sigma_term, None, None, None

    # Singular vectors that receive gradient. Pairs with no gradient on either vector contribute
    # exactly 0; their factors are zeroed up front so huge-but-finite F, G cannot overflow in F +- G
    # and give inf * 0 = nan.
    active = torch.zeros(k, dtype=torch.bool, device=sigma.device)
    if gu is not None:
        active = active | (gu != 0).any(dim=-2)
    if gvh is not None:
        active = active | (gvh != 0).any(dim=-1)
    eye = torch.eye(k, dtype=torch.bool, device=sigma.device)
    pair_active = ~eye & (active.unsqueeze(-2) | active.unsqueeze(-1))

    sigma_inv = _inverse(sigma, ~active)
    diff = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
    tot = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
    F = _inverse(diff, ~pair_active | (diff == 0))
    G = _inverse(tot, ~pair_active | (tot == 0))

    # Conservative check (see DEGENERACY_RTOL): any gradient into a singular vector whose pair is
    # numerically degenerate, or whose sigma, gap or sum has no usable reciprocal, makes the result
    # unreliable.
    near_degenerate = diff.abs() <= DEGENERACY_RTOL * sigma.max()
    if _KEPT_DEGENERATE == "zero":
        small = sigma <= ZERO_MAX_SIGMA * sigma.max()
        both = (near_degenerate & ~eye & active.unsqueeze(-2) & active.unsqueeze(-1)
                & small.unsqueeze(-2) & small.unsqueeze(-1))
        F = torch.where(both, torch.zeros_like(F), F)
        global _ZEROED_PAIRS
        _ZEROED_PAIRS += int(both.sum().item()) // 2
        checked = pair_active & ~both  # a zeroed pair is exempt, also when its gap is exactly 0
    else:
        checked = pair_active
    bad = (checked & (near_degenerate | _unsafe(diff) | _unsafe(tot))).any() or (_unsafe(sigma) & active).any()

    uh = u.conj().transpose(-2, -1)
    if gu is not None:
        guh = gu.conj().transpose(-2, -1)
        skew_u = uh @ gu - guh @ u
        u_term = u @ ((F + G).mul(skew_u)) * 0.5
        if m > k:
            proj_on_ortho_u = -u @ uh
            proj_on_ortho_u.diagonal(0, -2, -1).add_(1)
            u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2))
        u_term = u_term @ vh
    else:
        u_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)

    v = vh.conj().transpose(-2, -1)
    if gvh is not None:
        gv = gvh.conj().transpose(-2, -1)
        skew_v = vh @ gv - gvh @ v
        v_term = ((F - G).mul(skew_v)) @ vh * 0.5
        if n > k:
            proj_on_v_ortho = -v @ vh
            proj_on_v_ortho.diagonal(0, -2, -1).add_(1)
            v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
        v_term = u @ v_term
    else:
        v_term = torch.zeros(m, n, dtype=u.dtype, device=u.device)

    dA = u_term + sigma_term + v_term
    if u.is_complex() or v.is_complex():
        L = (uh @ gu).diagonal(0, -2, -1)
        L.real.zero_()
        L.imag.mul_(sigma_inv)
        dA = dA + (u * L.unsqueeze(-2)) @ vh
    if bad or not torch.isfinite(dA).all():
        warnings.warn(f"SVD backward: gradient reaches singular vectors with near-degenerate (gap <= {DEGENERACY_RTOL:g}"
                      " * sigma_max of the block), zero or underflowing singular values, or the result is not finite;"
                      " the result is unreliable.", DegenerateSVDWarning, stacklevel=2)
    return dA, None, None, None


def gap_safe_truncation(tol, D_total, margin=10 * DEGENERACY_RTOL, eps_multiplet=None):
    """CTM projector mask (``EnvCTM.iterate_(truncation_f=...)``) that cuts neither a multiplet nor a
    near-degenerate pair.

    The kept set is a prefix of the globally sorted spectrum (all blocks), starting at
    min(#values > tol * max, D_total); the cut moves down until (a) the relative gap between the last
    kept and first discarded value exceeds ``eps_multiplet`` (if given), so a multiplet spread over
    several symmetry sectors is kept or dropped whole, and (b) in every block the last kept and first
    discarded value differ by more than ``margin`` * (block max), so the SVD backward (which judges
    degeneracy per block, threshold DEGENERACY_RTOL) never sees a kept/discarded degenerate pair.
    Both are checked at the same cut, so neither undoes the other. Motivation (D = 4, chi = 32): a
    plain cut through a multiplet made CTM cycle, and tail values ~1e-8 sigma_max closer than
    1e-10 sigma_max straddled the tol cut and failed the gradient.
    """
    def mask_f(S):
        be = S.config.backend
        s = np.asarray(be.to_numpy(S.data), dtype=float)
        block = np.empty(len(s), dtype=int)
        bmax = []
        for b, sl in enumerate(S.slices):
            lo, hi = sl.slcs[0]
            block[lo:hi] = b
            bmax.append(s[lo:hi].max() if hi > lo else 0.0)
        order = np.argsort(-s, kind="stable")
        ss, bb = s[order], block[order]
        k = min(int(np.sum(ss > tol * ss[0])) if len(ss) else 0, D_total)

        def ok(k):
            if k == 0 or k == len(ss):
                return True
            if eps_multiplet is not None and ss[k - 1] - ss[k] <= eps_multiplet * ss[k - 1]:
                return False
            for b in set(bb[:k].tolist()) & set(bb[k:].tolist()):
                last_kept = ss[:k][bb[:k] == b].min()
                first_dropped = ss[k:][bb[k:] == b].max()
                if last_kept - first_dropped <= margin * bmax[b]:
                    return False
            return True

        while not ok(k):
            k -= 1
        mask = S.copy()
        mask._data = mask._data > float("inf")  # all False, backend-native bool
        if k:
            mask._data[order[:k].tolist()] = True  # list index: numpy and torch
        return mask
    return mask_f


def patch_svd_backward():
    """Install :func:`svd_backward` in place of yastn's SVD backward. Returns an undo function."""
    from yastn.backend.linalg import torch_svd_gesdd
    original = torch_svd_gesdd.SVDGESDD.backward
    torch_svd_gesdd.SVDGESDD.backward = staticmethod(svd_backward)

    def undo():
        torch_svd_gesdd.SVDGESDD.backward = staticmethod(original)
    return undo


@contextlib.contextmanager
def exact_svd_backward(kept_degenerate="warn"):
    """Context manager form of :func:`patch_svd_backward`; backward() must run inside it.

    ``kept_degenerate``: "warn" or "zero", see ``_KEPT_DEGENERATE``.
    """
    global _KEPT_DEGENERATE
    if kept_degenerate not in ("warn", "zero"):
        raise ValueError(kept_degenerate)
    undo, previous = patch_svd_backward(), _KEPT_DEGENERATE
    _KEPT_DEGENERATE = kept_degenerate
    try:
        yield
    finally:
        _KEPT_DEGENERATE = previous
        undo()


def peps_to_dict(psi):
    """``psi.to_dict()`` with a geometry entry that ``Peps.from_dict`` rebuilds correctly."""
    d = psi.to_dict()
    g = psi.geometry
    if isinstance(g, fpeps.TriangularLattice):
        d["geometry"] = {"type": "TriangularLattice", "dims": list(g.dims),
                         "boundary": g.boundary, "full_patch": g.full_patch}
    return d
