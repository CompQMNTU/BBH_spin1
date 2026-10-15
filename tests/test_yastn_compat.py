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
