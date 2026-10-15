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
    bad = (pair_active & (near_degenerate | _unsafe(diff) | _unsafe(tot))).any() or (_unsafe(sigma) & active).any()

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


def patch_svd_backward():
    """Install :func:`svd_backward` in place of yastn's SVD backward. Returns an undo function."""
    from yastn.backend.linalg import torch_svd_gesdd
    original = torch_svd_gesdd.SVDGESDD.backward
    torch_svd_gesdd.SVDGESDD.backward = staticmethod(svd_backward)

    def undo():
        torch_svd_gesdd.SVDGESDD.backward = staticmethod(original)
    return undo


@contextlib.contextmanager
def exact_svd_backward():
    """Context manager form of :func:`patch_svd_backward`; backward() must run inside it."""
    undo = patch_svd_backward()
    try:
        yield
    finally:
        undo()


def peps_to_dict(psi):
    """``psi.to_dict()`` with a geometry entry that ``Peps.from_dict`` rebuilds correctly."""
    d = psi.to_dict()
    g = psi.geometry
    if isinstance(g, fpeps.TriangularLattice):
        d["geometry"] = {"type": "TriangularLattice", "dims": list(g.dims),
                         "boundary": g.boundary, "full_patch": g.full_patch}
    return d
