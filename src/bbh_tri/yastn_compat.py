"""Workarounds for YASTN bugs at the pinned commit (docs/yastn_findings.md).

- SVD backward: ``yastn.backend.linalg.torch_svd_gesdd.safe_inverse`` ignores its ``eps_abs``
  argument and regularizes 1/x as x/(x^2 + 1e-12) with an *absolute* 1e-12. CTM matrices have
  singular values and gaps far below 1e-6, so AD gradients through ctmrg/fp_ctmrg come out wrong
  (2% at chi=8, 40% at chi=16 for a D=2 test state). ``exact_svd_backward`` swaps in the exact
  inverse (0 at exactly 0, matching the zeroed diagonal of F and G).
- Serialization: ``TriangularLattice.to_dict`` keeps only ``pattern``, so ``Peps.from_dict``
  rebuilds the default geometry (a full_patch or OBC state silently changes shape).
  ``peps_to_dict`` stores the constructor arguments instead.
"""

import contextlib

import yastn.tn.fpeps as fpeps


def _exact_inverse(x, eps_abs=None):
    import torch
    zero = x == 0
    return torch.where(zero, torch.zeros_like(x), 1.0 / torch.where(zero, torch.ones_like(x), x))


def patch_svd_backward():
    """Replace yastn's regularized inverse in the SVD backward by the exact one. Returns an undo function."""
    from yastn.backend.linalg import torch_svd_gesdd
    original = torch_svd_gesdd.safe_inverse
    torch_svd_gesdd.safe_inverse = _exact_inverse

    def undo():
        torch_svd_gesdd.safe_inverse = original
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
