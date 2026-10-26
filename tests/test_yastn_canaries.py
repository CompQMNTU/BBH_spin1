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

from yastn_utils import converged_ctm, spin1, ss_yastn, svd_directional_derivative

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


@pytest.mark.ad
def test_svd_backward_absolute_regularizer_is_wrong_for_small_singular_values():
    import numpy as np
    spectrum = 0.5 ** np.arange(9)
    ad, fd = svd_directional_derivative(spectrum, keep=6)
    assert ad == pytest.approx(fd, rel=1e-6)  # fine at O(1) scale
    ad, fd = svd_directional_derivative(1e-5 * spectrum, keep=6)
    assert abs(ad - fd) / abs(fd) > 0.5  # wrong once singular values drop below ~1e-6


@pytest.mark.parametrize("kw", [{"full_patch": True}, {"dims": (2, 3), "boundary": "obc", "full_patch": True}])
def test_triangular_to_dict_loses_geometry(ops, kw):
    psi = fpeps.product_peps(fpeps.TriangularLattice(**kw), ops.vec_z(val=1))
    g = fpeps.Peps.from_dict(psi.to_dict()).geometry
    assert (g.dims, g.boundary, g.full_patch) == ((3, 3), "infinite", False)  # constructor defaults


def test_obc_without_full_patch_has_three_sites():
    """boundary='obc' alone keeps the 3-site sublattice pattern; finite patches need full_patch=True."""
    assert len(fpeps.TriangularLattice(dims=(2, 3), boundary="obc").sites()) == 3


def test_full_patch_bonds_without_direction_raises():
    with pytest.raises(TypeError):
        fpeps.TriangularLattice(full_patch=True).bonds()


@pytest.mark.parametrize("initialization, D_after", [("EAT_SVD", 3), ("EAT", 1)])
def test_ntu_default_initialization_inflates_bonds_on_path_gates(initialization, D_after):
    """NTU truncation of a 3-site path gate (triangle MPO) on the exact FM eigenstate: the default
    'EAT_SVD' picks the SVD initialization, which keeps metric-null components (bond 1 -> 3);
    'EAT' keeps D = 1. Over hundreds of steps the junk drives the energy up (docs/yastn_findings.md)."""
    import numpy as np
    from bbh_tri import gates, lattice, operators, states
    from bbh_tri.hamiltonian import Model
    ops = operators.spin1("dense")
    geo = lattice.geometry("sqrt3")
    psi = states.product_peps(geo, states.seed_vectors("FM"), ops)
    tri = gates.nn_gates(geo, Model(0.7 * np.pi), ops, 0.005)[0]
    fpeps.evolution_step_(fpeps.EnvNTU(psi), [tri], opts_svd={"D_total": 4, "tol": 1e-12},
                          initialization=initialization)
    assert max(d for s in psi.sites() for d in psi[s].get_shape()[:4]) == D_after


@pytest.mark.ad
def test_fp_ctmrg_passes_truncation_f_to_forward_and_backward():
    """fp_ctmrg hands ctm_opts_fwd's truncation_f (as opts_svd['mask_f']) to the forward CTM, the
    gauge-fixing step and the backward fixed-point iteration, so one mask sets the cut everywhere
    (its multi-GPU forward, fwd_devices, drops it). docs/yastn_findings.md, U4c."""
    torch = pytest.importorskip("torch")
    from yastn.tn.fpeps.envs.fixed_pt import fp_ctmrg
    from bbh_tri.yastn_compat import exact_svd_backward
    ops = yastn.operators.Spin1(sym="dense", backend="torch", default_dtype="float64")
    geo = fpeps.TriangularLattice()
    ops.random_seed(seed=3)
    product = fpeps.product_peps(geo, ops.vec_z(val=1))
    params, tensors = [], {}
    for s in geo.sites():
        legs = product[s].get_legs()
        legs = [yastn.Leg(ops.config, s=leg.s, D=(2,)) for leg in legs[:4]] + list(legs[4:])
        t = yastn.rand(ops.config, legs=legs)
        params.append(t._data.clone().requires_grad_())
        t._data = params[-1] / params[-1].norm()
        tensors[s] = t
    calls = []

    def keep5(S):  # keep the 5 largest values (no gap is near-degenerate on this random state)
        calls.append(1)
        mask = S.copy()
        s = sorted(mask._data.detach().tolist(), reverse=True)
        mask._data = mask._data >= s[min(5, len(s)) - 1]
        return mask
    with exact_svd_backward():
        env = fp_ctmrg(fpeps.EnvCTM(fpeps.Peps(geo, tensors=tensors), init="eye"),
                       ctm_opts_fwd={"method": "2x2", "corner_tol": 1e-10, "max_sweeps": 300, "use_qr": False,
                                     "opts_svd": {"policy": "fullrank", "D_total": 8, "tol": 1e-8},
                                     "truncation_f": keep5},
                       ctm_opts_fp={"opts_svd": {"policy": "fullrank"}})
        assert all(env[s].tl.get_shape() == (5, 5) for s in geo.sites())  # forward: the mask, not D_total = 8
        n_forward = len(calls)
        sum(env[s].tl.norm() for s in geo.sites()).backward()
    assert len(calls) > n_forward  # backward fixed-point iteration
    assert all(torch.isfinite(p.grad).all() for p in params)
