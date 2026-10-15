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


def test_chain_classes():
    """Only 'h' lies along the horizontal chains; every NNN vector has a row component."""
    assert lattice.CHAIN_CLASSES == ("h",)


def test_anisotropy_default_is_isotropic():
    assert Model(0.3, J2=0.2, K2=-0.1).couplings() == Model(0.3, J2=0.2, K2=-0.1, J_anis=1.0).couplings()


def test_anisotropy_scales_interchain_bonds():
    """Niesen & Corboz Sec. IV F: J_anis scales 'v' and 'd'; here also all NNN classes (SPEC §0)."""
    a = 0.4
    c0, c = Model(0.3, J2=0.2, K2=-0.1).couplings(), Model(0.3, J2=0.2, K2=-0.1, J_anis=a).couplings()
    assert c["h"] == c0["h"]
    for cls in ("v", "d") + lattice.NNN_CLASSES:
        assert c[cls] == pytest.approx((a * c0[cls][0], a * c0[cls][1])), cls


@pytest.mark.parametrize("J2,K2", [(0.0, 0.0), (0.2, -0.1)])
def test_anisotropy_zero_decouples_chains(J2, K2):
    """J_anis = 0: decoupled horizontal chains, only the 'h' class remains, with or without NNN."""
    assert set(Model(0.3, J2=J2, K2=K2, J_anis=0.0).couplings()) == {"h"}


def test_anisotropic_bond_hamiltonians():
    ops = op.spin1("dense")
    m = Model(-0.1, J2=0.3, K2=0.05, J_anis=0.5)
    S = op.as_matrix(ops, op.ss(ops))
    for cls, h in m.bond_hamiltonians(ops).items():
        a = 1.0 if cls == "h" else 0.5
        J, K = (np.cos(-0.1), np.sin(-0.1)) if cls in lattice.NN_CLASSES else (0.3, 0.05)
        assert np.allclose(op.as_matrix(ops, h), a * (J * S + K * S @ S)), cls
