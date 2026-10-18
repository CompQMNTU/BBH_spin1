"""M4 AD driver (bbh_tri.variational, docs/ad_workflow.md).

Gradient gate (ad_workflow §5) on the hard case: a D = 3 state at theta = pi/4 whose CTM projector
SVDs keep an exactly degenerate pair (sigma_1 = sigma_2 ~ 2e-4 sigma_max, robust to 1e-2 noise on
the tensors). There:
- unrolled backpropagation through the CTM sweeps with kept-degenerate pairs zeroed
  (``kept_degenerate='zero'``) matches central finite differences to ~1e-8;
- fp_ctmrg's implicit fixed-point gradient is right on this state (3e-6) but was wrong by a factor
  ~10 on a fully converged D = 3 EnvBP state at the same theta (also with a kept degenerate pair;
  AD changed with corner_tol while FD did not), so the driver defaults to ad='unrolled';
- the default "warn" mode raises DegenerateSVDWarning (so the test exercises the degenerate path).
At D = 2 (no degeneracy) both AD modes agree with finite differences.
"""

import warnings

import numpy as np
import pytest
import yastn.tn.fpeps as fpeps

torch = pytest.importorskip("torch")

from bbh_tri import evolve, lattice, measure, operators, states, variational as var  # noqa: E402
from bbh_tri.hamiltonian import Model  # noqa: E402
from bbh_tri.yastn_compat import DegenerateSVDWarning, exact_svd_backward  # noqa: E402

pytestmark = pytest.mark.ad

THETA = np.pi / 4


def bp_state(D, seed=1, steps=40):
    """Short EnvBP run (tau = 0.1) from a random product state; numpy backend."""
    ops = operators.spin1("dense")
    psi = states.random_product_peps(lattice.geometry("sqrt3"), ops, seed=seed)
    evolve.evolve(psi, Model(THETA), ops, D, schedule=(0.1,), max_steps=steps)
    return psi


@pytest.fixture(scope="module")
def psi_d3():
    return bp_state(3)


def chain(psi, D, **kw):
    ops = var.torch_ops()
    return var.Chain(var.convert(psi, ops.config), Model(THETA), ops, chi=2 * D * D, **kw)


def directional(ch, kept_degenerate, eps=1e-5, seed=0):
    """(AD, central FD) derivative of the chain's loss along a fixed random direction."""
    with exact_svd_backward(kept_degenerate), warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateSVDWarning)
        E, _ = ch.loss()
        E.backward()
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g, dtype=p.dtype) for p in ch.params]
    ad = sum((p.grad * vi).sum() for p, vi in zip(ch.params, v)).item()
    x0 = [p.detach().clone() for p in ch.params]
    vals = []
    for sgn in (1, -1):
        with torch.no_grad():
            for p, q, vi in zip(ch.params, x0, v):
                p.copy_(q + sgn * eps * vi)
            vals.append(ch.loss()[0].item())
    with torch.no_grad():
        for p, q in zip(ch.params, x0):
            p.copy_(q)
    return ad, (vals[0] - vals[1]) / (2 * eps)


def test_convert_and_pad_keep_the_state():
    psi = bp_state(2, steps=20)
    ops = operators.spin1("dense")
    model = Model(THETA)
    E2 = measure.energy(psi, measure.ctm_env(psi, 8), ops, model)[0]
    back = var.convert(var.convert(psi, var.torch_ops().config), ops.config)
    assert measure.energy(back, measure.ctm_env(back, 8), ops, model)[0] == pytest.approx(E2, abs=1e-12)
    padded = var.pad(psi, 3, noise=0.0)
    assert {padded[s].get_shape()[:4] for s in padded.sites()} == {(3, 3, 3, 3)}
    assert measure.energy(padded, measure.ctm_env(padded, 8), ops, model)[0] == pytest.approx(E2, abs=1e-9)
    noisy = var.pad(psi, 3, noise=1e-3)
    assert measure.energy(noisy, measure.ctm_env(noisy, 8), ops, model)[0] == pytest.approx(E2, abs=1e-4)


def test_pad_keeps_complex_entries():
    psi = bp_state(2, steps=5)
    for s in psi.sites():
        psi[s] = 1j * psi[s]
    padded = var.pad(psi, 3, noise=0.0)
    for s in psi.sites():
        a, b = psi[s].to_numpy(), padded[s].to_numpy()
        assert np.array_equal(b[:2, :2, :2, :2], a) and not b[2:].any()


@pytest.mark.slow
@pytest.mark.parametrize("ad", ["unrolled", "fp"])
def test_gradient_matches_fd_without_degeneracy(ad):
    ad_val, fd = directional(chain(bp_state(2, steps=20), 2, ad=ad), "warn")
    assert ad_val == pytest.approx(fd, rel=1e-5)


@pytest.mark.slow
def test_degenerate_state_warns_by_default(psi_d3):
    with pytest.raises(DegenerateSVDWarning):
        directional(chain(psi_d3, 3), "warn")


@pytest.mark.slow
def test_unrolled_gradient_matches_fd_with_kept_degenerate_pair(psi_d3):
    ad_val, fd = directional(chain(psi_d3, 3, ad="unrolled"), "zero")
    assert ad_val == pytest.approx(fd, rel=1e-6)


def test_fd_check_catches_component_orthogonal_to_the_gradient():
    """Codex counterexample (PR #8): L = eps R[0,0] + R[1,2] + R[2,1] of the reconstruction of
    A = diag(1, 1e-4, 1e-4). The zero mode keeps only eps along A[0,0] and drops the O(1)
    off-diagonal part. The check along the gradient alone passes; random directions fail it."""
    import yastn
    cfg = yastn.make_config(backend="torch", default_dtype="float64")
    A = torch.diag(torch.tensor([1.0, 1e-4, 1e-4], dtype=torch.float64)).requires_grad_()

    def L():
        T = yastn.Tensor(config=cfg, s=(1, -1))
        T.set_block(val=A, Ds=(3, 3))
        U, S, V = yastn.linalg.svd(T, axes=(0, 1))
        R = (U @ S @ V).to_dense()
        return 1e-3 * R[0, 0] + R[1, 2] + R[2, 1]

    with exact_svd_backward("zero"):
        L().backward()
    assert var.fd_check(lambda: L().item(), [A], [A.grad], n_random=0)[0]  # the blind spot
    ok, checks, tol = var.fd_check(lambda: L().item(), [A], [A.grad], n_random=2)
    assert not ok


@pytest.mark.slow
def test_chain_fd_checks_gradient_after_zeroing(psi_d3, monkeypatch):
    """A step that zeroes a degenerate pair triggers check_gradient; a mismatch fails the step."""
    ch = chain(psi_d3, 3)
    entry = ch.step()
    assert entry["zeroed_pairs"] > 0
    assert len(entry["fd_check"]) == 3  # gradient + 2 random directions
    ch2 = chain(psi_d3, 3)
    monkeypatch.setattr(ch2, "check_gradient", lambda seed: (False, [(1.0, 2.0)], 0.0))
    with pytest.raises(DegenerateSVDWarning, match="FD check"):
        ch2.step()


@pytest.mark.slow
def test_chain_lowers_energy_and_resumes_deterministically(tmp_path):
    psi = bp_state(2, steps=20)
    ck = tmp_path / "chain.pt"
    ch = chain(psi, 2, checkpoint=str(ck))
    E0 = ch.loss()[0].item()
    for _ in range(3):
        ch.step()
    assert ch.history[-1]["energy"] < E0
    resumed = chain(psi, 2).load(str(ck))
    a, b = ch.step(), resumed.step()
    assert a["energy"] == b["energy"]
    assert all(torch.equal(p, q) for p, q in zip(ch.params, resumed.params))
    with pytest.raises(ValueError, match="checkpoint"):  # other chi
        var.Chain(var.convert(psi, var.torch_ops().config), Model(THETA), var.torch_ops(), chi=9).load(str(ck))
    with pytest.raises(ValueError, match="checkpoint"):  # other optimizer settings (reset by perturb)
        chain(psi, 2, lr=0.5).load(str(ck))
    with pytest.raises(ValueError, match="checkpoint"):  # other theta
        var.Chain(var.convert(psi, var.torch_ops().config), Model(0.3), var.torch_ops(), chi=8).load(str(ck))
    cell = fpeps.TriangularLattice(dims=(1, 3), full_patch=True)  # same tensor shapes, other unit cell
    other = fpeps.Peps(cell, tensors={s: psi[s] for s in cell.sites()})
    with pytest.raises(ValueError, match="checkpoint"):
        var.Chain(var.convert(other, var.torch_ops().config), Model(THETA), var.torch_ops(), chi=8).load(str(ck))


@pytest.mark.slow
def test_validate_reports_energy_per_chi():
    psi = bp_state(2, steps=20)
    out = var.validate(psi, Model(THETA), [8, 4])
    assert set(out["energies"]) == {4, 8}
    nn = [out["class_energies"][8][c] for c in ("h", "v", "d")]
    assert out["c3_spread"] == max(nn) - min(nn)  # diagnostics at the largest chi, not the last given
    assert abs(out["energies"][8] - out["energies"][4]) < 1e-2
    assert out["c3_spread"] >= 0 and len(out["order"]) == 3
