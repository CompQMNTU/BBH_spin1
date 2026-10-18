"""Spin-1 operators (SPEC §2.2), for dense, U1, U1xU1 and Z2xZ2 configs (SPEC §4.5).

Two-site operators follow ``yastn.fkron``: legs (out1, in1, out2, in2), signature (1, -1, 1, -1).
Dense and U1 use ``yastn.operators.Spin1`` and build everything from Sz, S+ and S- (U1 has no Sx,
Sy). U1xU1 and Z2xZ2 use :class:`SymSpin1`: there S+ has no definite charge, so every operator is
built from its matrix in the canonical basis, and an operator that is not charge-neutral (or of one
definite charge) raises.
"""

import numpy as np
import yastn
from yastn.sym.sym_abelian import sym_abelian

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)


class sym_Z2xZ2(sym_abelian):
    """Z2 x Z2, e.g. the pi rotations {exp(i pi Sx), exp(i pi Sz)} of an integer spin."""

    SYM_ID = "Z2xZ2"
    NSYM = 2

    @classmethod
    def fuse(cls, charges, signatures, new_signature):
        return np.mod(new_signature * (charges.swapaxes(1, 2) @ signatures), 2)


# Local bases of SymSpin1: (symmetry, [(charge, state in the canonical basis Sz = (+1, 0, -1))]),
# listed in ascending charge order, which is the order of ``Tensor.to_numpy``.
# U1xU1 (SU(3) point only): Cartan subgroup of SU(3), charges (Sz, Sz^2); every flavor count is
#   conserved by the SU(3)-invariant H at J = K, not elsewhere.
# Z2xZ2: Cartesian states |a> with S_a |a> = 0 (S_a |b> = i eps_abc |c>); charges are the parities
#   of exp(i pi Sx), exp(i pi Sz) (charge 1 = eigenvalue -1). H and the diagonal quadrupoles are real
#   in this basis.
_BASES = {
    "U1xU1": ("U1xU1", [((-1, 1), (0, 0, 1)), ((0, 0), (0, 1, 0)), ((1, 1), (1, 0, 0))]),
    "Z2xZ2": (sym_Z2xZ2, [((0, 1), (-1 / SQRT2, 0, 1 / SQRT2)),   # |x>
                          ((1, 0), (0, 1, 0)),                    # |z>
                          ((1, 1), (1j / SQRT2, 0, 1j / SQRT2))]),  # |y>
}
SYMS = ("dense", "U1") + tuple(_BASES)


class SymSpin1:
    """Spin-1 operators for the symmetries ``yastn.operators.Spin1`` lacks (U1xU1, Z2xZ2).

    Every basis state has its own charge (all blocks are 1x1). Operators come from matrices in the
    canonical basis via :meth:`operator`; ``sz``, ``sp``, ``sm`` exist only where they have a
    definite charge.
    """

    def __init__(self, sym, **config):
        sym_cls, basis = _BASES[sym]
        self.config = yastn.make_config(sym=sym_cls, **config)
        self.s = (1, -1)
        self.charges = [tuple(q) for q, _ in basis]
        self.basis = np.array([v for _, v in basis], dtype=complex).T  # columns: config basis states

    def random_seed(self, seed):
        self.config.backend.random_seed(seed)

    def space(self):
        return yastn.Leg(self.config, s=1, t=self.charges, D=(1,) * 3)

    def _charge(self, ts, s):
        a = np.array([[self.charges[i] for i in ts]], dtype=np.int64)
        return tuple(int(x) for x in self.config.sym.fuse(a, np.array(s), 1)[0])

    def operator(self, M, tol=1e-12):
        """Tensor of a 3^k x 3^k matrix M (canonical basis; rows (out1, out2, ...), columns (in1, ...)),
        legs (out1, in1, out2, in2, ...). Raises unless all nonzero elements carry one charge."""
        M = np.asarray(M, dtype=complex)
        k = int(round(np.log(M.shape[0]) / np.log(3)))
        U = self.basis
        Uk = U
        for _ in range(k - 1):
            Uk = np.kron(Uk, U)
        A = (Uk.conj().T @ M @ Uk).reshape((3,) * (2 * k))
        A = A.transpose([x for i in range(k) for x in (i, k + i)])  # (out1, in1, out2, in2, ...)
        if np.abs(A.imag).max() <= tol * max(np.abs(A).max(), 1):
            A = A.real
        s = self.s * k
        nz = [ix for ix in zip(*np.nonzero(np.abs(A) > tol))]
        ns = {self._charge(ix, s) for ix in nz}
        if len(ns) > 1:
            raise ValueError(f"operator mixes charges {sorted(ns)} in {self.config.sym.SYM_ID}")
        n = ns.pop() if ns else self.config.sym.zero()
        T = yastn.Tensor(config=self.config, s=s, n=n, dtype="complex128" if np.iscomplexobj(A) else "float64")
        for ix in nz:
            T.set_block(ts=[self.charges[i] for i in ix], Ds=(1,) * len(ix), val=A[ix].item())
        return T

    def charge_neutral(self, M, tol=1e-12):
        """True if the matrix M (canonical basis) is block-diagonal in this symmetry."""
        try:
            return self.operator(M, tol).struct.n == self.config.sym.zero()
        except ValueError:
            return False

    def vector(self, v):
        """Single-site state of the canonical-basis vector v; v must be a charge eigenstate (one config
        basis state up to a phase, which is dropped)."""
        w = self.basis.conj().T @ np.asarray(v, dtype=complex)
        nz = np.nonzero(np.abs(w) > 1e-12)[0]
        if len(nz) != 1:
            raise ValueError(f"state is not a {self.config.sym.SYM_ID} charge eigenstate")
        k = nz[0]
        vec = yastn.Tensor(config=self.config, s=(1,), n=self.charges[k])
        vec.set_block(ts=(self.charges[k],), Ds=(1,), val=abs(w[k]))
        return vec

    def I(self):
        return self.operator(np.eye(3))

    def _dense(self, name):
        return self.operator(as_matrix(_DENSE, getattr(_DENSE, name)()))

    def sz(self):
        return self._dense("sz")

    def sp(self):
        return self._dense("sp")

    def sm(self):
        return self._dense("sm")


def spin1(sym="dense", **config):
    """Spin-1 operator set for ``sym`` in :data:`SYMS`; extra keyword arguments (backend,
    default_dtype, ...) go to the config. Dense and U1: ``yastn.operators.Spin1``; else :class:`SymSpin1`."""
    if sym in _BASES:
        return SymSpin1(sym, **config)
    return yastn.operators.Spin1(sym=sym, **config)


def is_matrix_built(ops):
    return isinstance(ops, SymSpin1)


def is_dense(ops):
    return ops.config.sym.SYM_ID == "dense"


def ss(ops):
    """S_i·S_j = Sz Sz + (S+ S- + S- S+)/2 as a two-site operator (raises in U1xU1, where only
    J S·S + K (S·S)^2 with J = K is symmetric)."""
    if is_matrix_built(ops):
        return ops.operator(as_matrix(_DENSE, ss(_DENSE)))
    Sz, Sp, Sm = ops.sz(), ops.sp(), ops.sm()
    return yastn.fkron(Sz, Sz) + 0.5 * (yastn.fkron(Sp, Sm) + yastn.fkron(Sm, Sp)) + 0 * yastn.fkron(ops.I(), ops.I())


def two_site_product(A, B):
    """Operator product A·B of two-site operators (A acts after B)."""
    return yastn.tensordot(A, B, axes=((1, 3), (0, 2))).transpose(axes=(0, 2, 1, 3))


def ss2(ops):
    """(S_i·S_j)^2 as the operator product of S·S with itself."""
    S = ss(ops)
    return two_site_product(S, S)


def h_bond(ops, J, K):
    """Bond Hamiltonian J S·S + K (S·S)^2."""
    if is_matrix_built(ops):
        return ops.operator(as_matrix(_DENSE, h_bond(_DENSE, J, K)))
    S = ss(ops)
    return J * S + K * two_site_product(S, S)


def quadrupole(ops):
    """Traceless quadrupole components (SPEC §2.2), as single-site operators.

    Q = (Sx^2 - Sy^2, (3Sz^2 - 2)/sqrt3, {Sx,Sy}, {Sy,Sz}, {Sz,Sx}), keys 'Q1'..'Q5', with S(S+1) = 2.
    Dense: all five ({Sx,Sy} and {Sy,Sz} are complex). U1: only 'Q2'; the other components change
    Sz and have zero expectation in a U1-symmetric state, so |<Q>| = |<Q2>| there. U1xU1, Z2xZ2:
    the charge-neutral components only (U1xU1: Q2; Z2xZ2: Q1, Q2), for the same reason.
    """
    if is_matrix_built(ops):
        return _neutral(ops, quadrupole(_DENSE))
    Sz, I = ops.sz(), ops.I()
    out = {"Q2": (3 * (Sz @ Sz) - 2 * I) / SQRT3}
    if not is_dense(ops):
        return out
    Sp, Sm = ops.sp(), ops.sm()
    out["Q1"] = 0.5 * (Sp @ Sp + Sm @ Sm)                     # Sx^2 - Sy^2
    out["Q3"] = -0.5j * (Sp @ Sp - Sm @ Sm)                   # {Sx, Sy}
    out["Q4"] = -0.5j * (Sp @ Sz + Sz @ Sp - Sm @ Sz - Sz @ Sm)  # {Sy, Sz}
    out["Q5"] = 0.5 * (Sz @ Sp + Sp @ Sz + Sz @ Sm + Sm @ Sz)  # {Sz, Sx}
    return {k: out[k] for k in ("Q1", "Q2", "Q3", "Q4", "Q5")}


def spin_vector(ops):
    """Single-site spin components for the moment m = |<S>|: dense Sx, Sy, Sz; otherwise the
    charge-neutral ones (U1, U1xU1: Sz; Z2xZ2: none, m = 0 by symmetry)."""
    if is_matrix_built(ops):
        return _neutral(ops, spin_vector(_DENSE))
    if is_dense(ops):
        return {"Sx": ops.sx(), "Sy": ops.sy(), "Sz": ops.sz()}
    return {"Sz": ops.sz()}


def vector_norm(expectations):
    """|v| from a dict of component expectation values (missing components count as 0)."""
    return float(np.sqrt(sum(abs(x) ** 2 for x in expectations.values())))


def _neutral(ops, dense_ops):
    """{name: op in ``ops``} for the charge-neutral entries of a dict of dense single-site operators."""
    mats = {k: as_matrix(_DENSE, o) for k, o in dense_ops.items()}
    return {k: ops.operator(M) for k, M in mats.items() if ops.charge_neutral(M)}


def two_site_terms(ops, op2):
    """[(1, A, B)] with op2 = sum A ⊗ B and single-site A, B of definite charge (basis-transition
    operators |a><b| ⊗ |c><d| of the nonzero elements); for :func:`measure.bond_energy` on classes
    without an rdm."""
    M = as_matrix(ops, op2).reshape(3, 3, 3, 3)  # (out1, out2, in1, in2), canonical basis
    U = ops.basis if is_matrix_built(ops) else np.eye(3)
    M = np.einsum("ia,jb,ijkl,kc,ld->abcd", U.conj(), U.conj(), M, U, U)  # config basis
    out = []
    for a, b, c, d in zip(*np.nonzero(np.abs(M) > 1e-12)):
        E1, E2 = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
        E1[a, c], E2[b, d] = M[a, b, c, d], 1
        out.append((1.0, ops.operator(U @ E1 @ U.conj().T), ops.operator(U @ E2 @ U.conj().T)))
    return out


def _full_numpy(ops, op):
    """``op.to_numpy()`` on full physical legs. A charged U1 operator (e.g. S+) stores only the
    sectors it connects, so a bare ``to_numpy()`` would return a smaller array."""
    leg = ops.space()
    return op.to_numpy(legs={i: leg if s == leg.s else leg.conj() for i, s in enumerate(op.s)})


def basis_permutation(ops):
    """Permutation taking the config's basis to the canonical order Sz = (+1, 0, -1).

    The dense config already uses it; U1's ``to_numpy`` orders charges (-1, 0, +1).
    """
    return np.argsort(-np.real(np.diag(_full_numpy(ops, ops.sz()))), kind="stable")


def as_matrix(ops, op):
    """Single-site (3x3) or two-site (9x9, rows (out1, out2), columns (in1, in2)) operator as a numpy
    matrix in the canonical basis Sz = (+1, 0, -1)."""
    a = _full_numpy(ops, op)
    if is_matrix_built(ops):  # config basis -> canonical basis
        U = ops.basis
        if a.ndim == 2:
            return U @ a @ U.conj().T
        if a.ndim == 4:
            U2 = np.kron(U, U)
            return U2 @ a.transpose(0, 2, 1, 3).reshape(9, 9) @ U2.conj().T
        raise ValueError(f"expected a one- or two-site operator, got {a.ndim} legs")
    p = basis_permutation(ops)
    if a.ndim == 2:
        return a[np.ix_(p, p)]
    if a.ndim == 4:
        a = a[np.ix_(p, p, p, p)]
        return a.transpose(0, 2, 1, 3).reshape(9, 9)
    raise ValueError(f"expected a one- or two-site operator, got {a.ndim} legs")


def basis_matrix(ops):
    """3x3 matrix whose columns are the config's basis states (``to_numpy`` order) in the canonical
    basis Sz = (+1, 0, -1)."""
    if is_matrix_built(ops):
        return ops.basis
    B = np.zeros((3, 3))
    B[np.arange(3), basis_permutation(ops)] = 1
    return B


def state_vector(ops, v):
    """Single-site yastn state of the canonical-basis vector v (in U1 and the SymSpin1 symmetries
    it must be a charge eigenstate)."""
    if is_matrix_built(ops):
        return ops.vector(v)
    sz_of = {0: 1, 1: 0, 2: -1}
    v = np.asarray(v)
    nz = [k for k in range(3) if v[k] != 0]
    if not is_dense(ops) and len(nz) != 1:
        raise ValueError("U1 product states must be Sz eigenstates")
    terms = [v[k].item() * ops.vec_z(val=sz_of[k]) for k in nz]  # python scalars: numpy * Tensor misbehaves
    vec = terms[0]
    for t in terms[1:]:
        vec = vec + t
    return vec


_DENSE = spin1("dense")
