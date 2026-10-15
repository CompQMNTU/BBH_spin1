"""M1: model couplings and bond Hamiltonians (SPEC §0, §2.3)."""

import numpy as np
import pytest

from bbh_tri import lattice
from bbh_tri import operators as op
from bbh_tri.hamiltonian import Model, energy_per_site


def test_nn_only_couplings():
    m = Model(theta=0.3)
    assert set(m.couplings()) == set(lattice.NN_CLASSES)
    assert all(jk == (np.cos(0.3), np.sin(0.3)) for jk in m.couplings().values())


def test_nnn_couplings():
    m = Model(theta=0.3, J2=0.2, K2=-0.1)
    c = m.couplings()
    assert set(c) == set(lattice.NN_CLASSES + lattice.NNN_CLASSES)
    assert all(c[k] == (0.2, -0.1) for k in lattice.NNN_CLASSES)


@pytest.mark.parametrize("sym", ["dense", "U1"])
def test_bond_hamiltonians(sym):
    ops = op.spin1(sym)
    m = Model(theta=1.1, J2=0.4, K2=0.25)
    S = op.as_matrix(ops, op.ss(ops))
    for cls, h in m.bond_hamiltonians(ops).items():
        J, K = m.couplings()[cls]
        assert np.allclose(op.as_matrix(ops, h), J * S + K * S @ S), cls


def test_su3_point_is_swap():
    """At theta = pi/4, h = (P_swap + 1)/sqrt2 (operator identity; docs/yastn_findings.md SU(3) sanity run)."""
    ops = op.spin1("dense")
    h = op.as_matrix(ops, Model(np.pi / 4).bond_hamiltonians(ops)["h"])
    P = np.eye(9)[[3 * (k % 3) + k // 3 for k in range(9)]]
    assert np.allclose(h, (P + np.eye(9)) / np.sqrt(2))


def test_energy_per_site():
    assert energy_per_site({"h": 0.5, "v": 0.25, "d": -0.25}) == pytest.approx(0.5)


@pytest.mark.ad
def test_energy_per_site_keeps_autograd():
    """Codex review of PR #5: the aggregate must stay a differentiable torch scalar (AD loss)."""
    torch = pytest.importorskip("torch")
    x = torch.tensor([0.5, 0.25, -0.25], dtype=torch.float64, requires_grad=True)
    e = energy_per_site({"h": x[0], "v": x[1], "d": x[2] * 2})
    e.backward()
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64))
