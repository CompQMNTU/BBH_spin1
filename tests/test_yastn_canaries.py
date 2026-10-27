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
    """Upstream TriangularLattice.to_dict (bbh_tri.yastn_compat replaces it on import) keeps only the
    site pattern; from_dict then rebuilds the constructor defaults."""
    from bbh_tri.yastn_compat import _UPSTREAM_TRIANGULAR_TO_DICT
    psi = fpeps.product_peps(fpeps.TriangularLattice(**kw), ops.vec_z(val=1))
    d = {**psi.to_dict(), "geometry": _UPSTREAM_TRIANGULAR_TO_DICT(psi.geometry)}
    g = fpeps.Peps.from_dict(d).geometry
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


@pytest.mark.ad
@pytest.mark.slow
def test_fp_ctmrg_gradient_wrong_on_patch_cells():
    """fp_ctmrg's implicit gradient misses finite differences by O(1) on full_patch cells, also for
    the NN model and with the geometry-preserving to_dict (unrolled AD is exact there);
    variational.Chain refuses ad='fp' on them. docs/yastn_findings.md, U4d."""
    pytest.importorskip("torch")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_variational import directional
    from bbh_tri import evolve, lattice, operators, states, variational as var
    from bbh_tri.hamiltonian import Model
    ops = operators.spin1("dense")
    psi = states.random_product_peps(lattice.geometry("patch", (2, 2)), ops, seed=1)
    evolve.evolve(psi, Model(0.1), ops, 2, schedule=(0.1,), max_steps=20)
    tops = var.torch_ops()
    ch = var.Chain(var.convert(psi, tops.config), Model(0.1), tops, chi=8)
    ch.ad = "fp"
    ad, fd = directional(ch, "zero")
    assert abs(ad - fd) > 0.5 * abs(fd)  # measured: AD -0.0201, FD -0.2380


def test_path_gate_through_one_tensor_twice_runs_silently():
    """On the sqrt3 cell an NNN path starts and ends on the same unit-cell tensor; evolution_step_
    applies the gate without an error, though no correct result is possible. lattice.gate_paths
    drops such paths (docs/yastn_findings.md, U1b)."""
    from bbh_tri import lattice, operators, states
    ops3 = operators.spin1("dense")
    geo = lattice.geometry("sqrt3")
    path = lattice.paths(geo, lattice.bonds(geo, "nnn_d")[0])[0]
    assert len({geo.site2index(s) for s in path}) < len(path)
    psi = states.random_product_peps(geo, ops3, seed=1)
    g = fpeps.gates.gate_nn_exp(0.1, ops3.I(), operators.h_bond(ops3, 0.1, 0.0))._replace(sites=path)
    fpeps.evolution_step_(fpeps.EnvBP(psi), [g], opts_svd={"D_total": 4, "tol": 1e-12})  # no error


def _ctm_fu_setup(dims, D, chi, steps, J2=0.0):
    """A BP-evolved D state on a patch cell (sqrt3 if dims is None), its converged EnvCTM, and the
    first-order model gates (3-site MPOs: NN triangles, routed NNN)."""
    from bbh_tri import evolve, gates, lattice, operators, states
    from bbh_tri.hamiltonian import Model
    ops3 = operators.spin1("dense")
    geo = lattice.geometry("sqrt3") if dims is None else lattice.geometry("patch", dims)
    model = Model(0.0, J2=J2)
    psi = states.random_product_peps(geo, ops3, seed=1)
    evolve.evolve(psi, model, ops3, D, schedule=(0.1,), max_steps=steps, check_every=steps)
    return psi, converged_ctm(psi, chi, max_sweeps=30, corner_tol=1e-9), gates.model_gates(geo, model, ops3, 0.05, 0)


def test_ctm_full_update_of_mpo_gate_raises():
    """Full update (EnvCTM metric) of a 3-site MPO gate with the default method='mpo': the whole
    gate enlarges both path bonds, and bond_metric cannot contract the still-enlarged (hard-fused)
    neighbouring bond (docs/yastn_findings.md, full update)."""
    psi, env, gl = _ctm_fu_setup(None, 2, 4, 3)
    with pytest.raises(yastn.YastnError, match="hard-fused"):
        fpeps.evolution_step_(env, gl[:1], opts_svd={"D_total": 2, "tol": 1e-12})


def test_ctm_full_update_split_gates_need_post_truncation_opts():
    """method='NN' with EnvCTM: truncate_ always calls post_truncation_ -> update_bond_, which reads
    env.opts_svd unless opts_post_truncation passes opts_svd; EnvCTM never sets it."""
    psi, env, gl = _ctm_fu_setup(None, 2, 4, 3)
    with pytest.raises(AttributeError, match="opts_svd"):
        fpeps.evolution_step_(env, gl[:1], opts_svd={"D_total": 2, "tol": 1e-12}, method="NN")


def test_split_gate_2site_enlarges_the_physical_leg():
    """split_gate_2site SVDs the middle MPO tensor across (in, left) | (out, right): the pieces
    carry the 9-dim MPO bond on the middle site's physical leg (3 -> 27) between the two pieces."""
    from yastn.tn.fpeps._evolution import split_gate_2site
    from bbh_tri import gates, lattice, operators
    from bbh_tri.hamiltonian import Model
    ops3 = operators.spin1("dense")
    g = gates.model_gates(lattice.geometry("sqrt3"), Model(0.0), ops3, 0.05, 0)[0]
    assert [t.get_shape() for t in g.G] == [(3, 3, 9), (3, 3, 9, 9), (3, 3, 9)]
    a, b = split_gate_2site(g)
    assert [t.get_shape() for t in a.G] == [(3, 3, 9), (27, 3, 9)]
    assert [t.get_shape() for t in b.G] == [(3, 27, 9), (3, 3, 9)]


@pytest.mark.slow
def test_ctm_full_update_of_split_gates_fails_at_d3():
    """With method='NN' and opts_post_truncation the split gates run at D = 2 but not at D = 3:
    on the (3, 3) cell (theta = 0, J2 = 0.1, chi = 9) the 9th gate's EAT initialization finds no
    metric eigenvalue above the cutoff (ValueError); initialization='SVD' fails with
    ZeroDivisionError instead (zero metric norm of the state)."""
    import warnings
    psi, env, gl = _ctm_fu_setup((3, 3), 3, 9, 6, J2=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="zero-size array"):
            for g in gl:
                fpeps.evolution_step_(env, [g], opts_svd={"D_total": 3, "tol": 1e-12}, method="NN",
                                      opts_post_truncation={"opts_svd": {"D_total": 9, "tol": 1e-14}})
