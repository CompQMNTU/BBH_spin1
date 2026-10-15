"""M1: spin-1 operators in dense and U1 configs (SPEC §2.2)."""

import numpy as np
import pytest
import yastn

from bbh_tri import operators as op

SYMS = ["dense", "U1"]


@pytest.fixture(scope="module", params=SYMS)
def ops(request):
    return op.spin1(request.param)


@pytest.fixture(scope="module")
def dense():
    return op.spin1("dense")


def test_canonical_basis(ops):
    assert np.allclose(np.diag(op.as_matrix(ops, ops.sz())), [1, 0, -1])


def test_casimir(ops):
    """S^2 = Sz^2 + (S+S- + S-S+)/2 = s(s+1) = 2."""
    Sz, Sp, Sm = ops.sz(), ops.sp(), ops.sm()
    S2 = Sz @ Sz + 0.5 * (Sp @ Sm + Sm @ Sp)
    assert np.allclose(op.as_matrix(ops, S2), 2 * np.eye(3))


def test_ss_spectrum(ops):
    """S·S on two spin-1: [S_tot(S_tot+1) - 4]/2 = -2 (x1), -1 (x3), 1 (x5)."""
    ev = np.linalg.eigvalsh(op.as_matrix(ops, op.ss(ops)))
    assert np.allclose(ev, [-2] + [-1] * 3 + [1] * 5)


def test_ss2_is_matrix_square(ops):
    M = op.as_matrix(ops, op.ss(ops))
    assert np.allclose(op.as_matrix(ops, op.ss2(ops)), M @ M)


def test_h_bond(ops):
    J, K = 0.3, -1.7
    M = op.as_matrix(ops, op.ss(ops))
    assert np.allclose(op.as_matrix(ops, op.h_bond(ops, J, K)), J * M + K * M @ M)


def test_u1_matches_dense(dense):
    u1 = op.spin1("U1")
    for f in (op.ss, op.ss2):
        assert np.allclose(op.as_matrix(u1, f(u1)), op.as_matrix(dense, f(dense)))
    assert np.allclose(op.as_matrix(u1, op.quadrupole(u1)["Q2"]), op.as_matrix(dense, op.quadrupole(dense)["Q2"]))
    assert set(op.quadrupole(u1)) == {"Q2"} and set(op.spin_vector(u1)) == {"Sz"}


def test_two_site_ops_are_symmetric_charge_zero(ops):
    for f in (op.ss, op.ss2):
        assert f(ops).struct.n == ops.I().struct.n


def test_quadrupole_traceless_hermitian(dense):
    Q = op.quadrupole(dense)
    assert list(Q) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for q in Q.values():
        m = op.as_matrix(dense, q)
        assert abs(np.trace(m)) < 1e-14
        assert np.allclose(m, m.conj().T)


def test_quadrupole_matches_cartesian_definition(dense):
    """Q from S+- equals the Cartesian definition with Sx, Sy (SPEC §2.2)."""
    Sx, Sy, Sz = (op.as_matrix(dense, t) for t in (dense.sx(), dense.sy(), dense.sz()))
    ref = {"Q1": Sx @ Sx - Sy @ Sy, "Q2": (2 * Sz @ Sz - Sx @ Sx - Sy @ Sy) / np.sqrt(3), "Q3": Sx @ Sy + Sy @ Sx,
           "Q4": Sy @ Sz + Sz @ Sy, "Q5": Sz @ Sx + Sx @ Sz}
    for k, q in op.quadrupole(dense).items():
        assert np.allclose(op.as_matrix(dense, q), ref[k]), k


def test_qq_identity(dense):
    """Q_i·Q_j = 2 (S_i·S_j)^2 + S_i·S_j - 8/3 exactly (SPEC §2.2)."""
    QQ = sum(np.kron(m, m) for m in (op.as_matrix(dense, q) for q in op.quadrupole(dense).values()))
    M = op.as_matrix(dense, op.ss(dense))
    assert np.allclose(QQ, 2 * M @ M + M - 8 / 3 * np.eye(9))


@pytest.mark.parametrize("sz,q_norm", [(0, 2 / np.sqrt(3)), (1, 1 / np.sqrt(3)), (-1, 1 / np.sqrt(3))])
def test_q_norm_of_sz_states(ops, sz, q_norm):
    """|Q|(|0>) = 2/sqrt3, |Q|(|+-1>) = 1/sqrt3 (SPEC §2.2, §3); in U1 via the Q2 branch only."""
    v = np.zeros(3)
    v[[1, 0, -1].index(sz)] = 1  # canonical basis (+1, 0, -1)
    exp = {k: v @ op.as_matrix(ops, q) @ v for k, q in op.quadrupole(ops).items()}
    assert op.vector_norm(exp) == pytest.approx(q_norm, abs=1e-14)


def test_spin_vector_moment(dense):
    """m = |<S>| = 1 for a spin-coherent state along x, using the dense components."""
    v = np.array([1, np.sqrt(2), 1]) / 2  # |Sx = +1> in basis (+1, 0, -1)
    exp = {k: (v.conj() @ op.as_matrix(dense, s) @ v).real for k, s in op.spin_vector(dense).items()}
    assert exp["Sx"] == pytest.approx(1) and op.vector_norm(exp) == pytest.approx(1)


def test_as_matrix_charged_single_site_operators(dense):
    """Codex review of PR #5: U1 S+ stores only the sectors it connects; as_matrix must still give the
    full 3x3 matrix and agree with dense."""
    u1 = op.spin1("U1")
    for name in ("sp", "sm", "sz", "I"):
        assert np.allclose(op.as_matrix(u1, getattr(u1, name)()), op.as_matrix(dense, getattr(dense, name)())), name


def test_as_matrix_charged_two_site_operator(dense):
    u1 = op.spin1("U1")
    assert np.allclose(op.as_matrix(u1, yastn.fkron(u1.sp(), u1.sz())), op.as_matrix(dense, yastn.fkron(dense.sp(), dense.sz())))
