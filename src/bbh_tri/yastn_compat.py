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

# Pairs with |sigma_i - sigma_j| <= DEGENERACY_RTOL * sigma_max are numerically degenerate. For a
# gauge-invariant loss their 1/(sigma_i - sigma_j) term is 0/0 with a finite, nonzero limit that
# roundoff destroys; neither zeroing nor broadening recovers it (kept pair at gap 0..1e-14: 1e-4 to
# 80% gradient error; at 1e-12: up to 6e-4; at >= 1e-9: <= 1e-7). If such pairs carry more than
# DEGENERACY_WARN_FRACTION of the singular-vector gradient, svd_backward warns (false positives
# possible for losses that depend on a degenerate pair only through its span, e.g. a projector). CTM tails
# (sigma ~ 1e-20, gaps ~ 0) are flagged but carry no gradient: no warning at chi = 4-16.
DEGENERACY_RTOL = 1e-10
DEGENERACY_WARN_FRACTION = 1e-6


class DegenerateSVDWarning(RuntimeWarning):
    """Near-degenerate singular values contribute to an SVD gradient, which is then unreliable."""


def _inverse(x, zero):
    import torch
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

    eye = torch.eye(k, dtype=torch.bool, device=sigma.device)
    sigma_inv = _inverse(sigma, sigma == 0)
    diff = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
    F = _inverse(diff, eye | (diff == 0))
    near_degenerate = ~eye & (diff.abs() <= DEGENERACY_RTOL * sigma.max())
    tot = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
    G = _inverse(tot, eye | (tot == 0))

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

    if near_degenerate.any():
        # Within a numerically degenerate pair both the skew numerators and the gap vanish (0/0),
        # and the skew parts can cancel exactly (e.g. a symmetric incoming gradient), so they cannot
        # be used to detect the problem. Flag any gradient reaching the pair's singular vectors:
        # off-diagonal entries of U^H gU and V^H gV, which do not cancel.
        weight = torch.zeros_like(F)
        if gu is not None:
            ugu = (uh @ gu).abs()
            weight = weight + ugu + ugu.transpose(-2, -1)
        if gvh is not None:
            vgv = (vh @ gv).abs()
            weight = weight + vgv + vgv.transpose(-2, -1)
        weight = weight.masked_fill(eye, 0)
        total = weight.sum()
        if total > 0 and weight[near_degenerate].sum() > DEGENERACY_WARN_FRACTION * total:
            warnings.warn(f"SVD backward: near-degenerate singular values (gap <= {DEGENERACY_RTOL:g} * sigma_max of the "
                          "block) carry gradient; the result is unreliable.", DegenerateSVDWarning, stacklevel=2)

    dA = u_term + sigma_term + v_term
    if u.is_complex() or v.is_complex():
        L = (uh @ gu).diagonal(0, -2, -1)
        L.real.zero_()
        L.imag.mul_(sigma_inv)
        dA = dA + (u * L.unsqueeze(-2)) @ vh
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
