"""U4 (SPEC §1): fp_ctmrg (implicit-AD CTM) on TriangularLattice.

Finding (docs/yastn_findings.md, U4): fp_ctmrg runs on TriangularLattice, but its gradient
does NOT match finite differences: ~2% off at chi=8, ~40% off at chi>=16. Plain backprop
through ctmrg_ gives the same AD value at chi=8, and RectangularUnitcell with the same
pattern gives identical numbers, so the error is in differentiating the CTM step itself
(SVD backward under truncation / near-zero singular values), not in the fixed-point series
or the triangular geometry. Blocks the gradient gate of docs/ad_workflow.md §5.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import yastn  # noqa: E402
import yastn.tn.fpeps as fpeps  # noqa: E402
from yastn.tn.fpeps import Site  # noqa: E402
from yastn.tn.fpeps.envs.rdm import rdm2x2_anti_diagonal  # noqa: E402

pytestmark = pytest.mark.ad


def setup_problem(D=2, seed=3):
    from yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg

    ops = yastn.operators.Spin1(sym="dense", backend="torch", default_dtype="float64")
    geo = fpeps.TriangularLattice()
    ops.random_seed(seed=seed)
    product = fpeps.product_peps(geo, ops.vec_z(val=1))  # legs (t, l, b, r, phys), all D=1
    template = {}
    for s in geo.sites():
        legs = product[s].get_legs()
        legs = [yastn.Leg(ops.config, s=leg.s, D=(D,)) for leg in legs[:4]] + list(legs[4:])
        template[s] = yastn.rand(ops.config, legs=legs)
    Sz = ops.sz().to_dense()

    def loss(params, chi):
        tensors = {}
        for s, p in zip(geo.sites(), params):
            t = template[s].clone()
            t._data = p / p.norm()
            tensors[s] = t
        psi = fpeps.Peps(geo, tensors=tensors)
        env = fp_ctmrg(fpeps.EnvCTM(psi, init="eye"),
                       ctm_opts_fwd={"method": "2x2", "corner_tol": 1e-10, "max_sweeps": 300, "use_qr": False,
                                     "opts_svd": {"policy": "fullrank", "D_total": chi, "tol": 1e-8}},
                       ctm_opts_fp={"opts_svd": {"policy": "fullrank"}})
        out = 0
        for b in geo.bonds("d"):  # sum of <Sz Sz> over diagonal bonds
            r = rdm2x2_anti_diagonal(Site(b[1][0], b[0][1]), psi, env)[0].to_dense()
            out = out + torch.einsum("ij,jilk,kl->", Sz, r, Sz) / torch.einsum("iikk->", r)
        return out, env

    params = [template[s]._data.clone().requires_grad_() for s in geo.sites()]
    return loss, params


def test_fp_ctmrg_runs_on_triangular_lattice():
    loss, params = setup_problem()
    value, env = loss(params, chi=8)
    assert isinstance(env, fpeps.EnvCTM)  # returns env only, not the documented tuple
    value.backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in params)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="AD gradient off by ~2% vs finite differences (U4, docs/yastn_findings.md)")
def test_fp_ctmrg_gradient_matches_finite_difference():
    loss, params = setup_problem()
    chi, eps = 8, 1e-4
    value, _ = loss(params, chi)
    value.backward()
    v = [torch.randn(p.shape, generator=torch.Generator().manual_seed(0), dtype=p.dtype) for p in params]
    ad = sum((p.grad * vi).sum() for p, vi in zip(params, v)).item()
    with torch.no_grad():
        lp = loss([p + eps * vi for p, vi in zip(params, v)], chi)[0].item()
        lm = loss([p - eps * vi for p, vi in zip(params, v)], chi)[0].item()
    fd = (lp - lm) / (2 * eps)
    assert ad == pytest.approx(fd, rel=1e-4)
