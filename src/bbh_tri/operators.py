"""Spin-1 operators (SPEC §2.2), for dense and U1 configs.

Two-site operators follow ``yastn.fkron``: legs (out1, in1, out2, in2), signature (1, -1, 1, -1).
Everything is built from Sz, S+ and S- (U1 has no Sx, Sy).
"""

import numpy as np
import yastn

SQRT3 = np.sqrt(3)


def spin1(sym="dense", **config):
    """``yastn.operators.Spin1``; extra keyword arguments (backend, default_dtype, ...) go to the config."""
    return yastn.operators.Spin1(sym=sym, **config)


def is_dense(ops):
    return ops.config.sym.SYM_ID == "dense"


def ss(ops):
    """S_i·S_j = Sz Sz + (S+ S- + S- S+)/2 as a two-site operator."""
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
    S = ss(ops)
    return J * S + K * two_site_product(S, S)


def quadrupole(ops):
    """Traceless quadrupole components (SPEC §2.2), as single-site operators.

    Q = (Sx^2 - Sy^2, (3Sz^2 - 2)/sqrt3, {Sx,Sy}, {Sy,Sz}, {Sz,Sx}), keys 'Q1'..'Q5', with S(S+1) = 2.
    Dense: all five ({Sx,Sy} and {Sy,Sz} are complex). U1: only 'Q2'; the other components change
    Sz and have zero expectation in a U1-symmetric state, so |<Q>| = |<Q2>| there.
    """
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
    """Single-site spin components for the moment m = |<S>|: dense Sx, Sy, Sz; U1 only Sz."""
    if is_dense(ops):
        return {"Sx": ops.sx(), "Sy": ops.sy(), "Sz": ops.sz()}
    return {"Sz": ops.sz()}


def vector_norm(expectations):
    """|v| from a dict of component expectation values (missing components count as 0)."""
    return float(np.sqrt(sum(abs(x) ** 2 for x in expectations.values())))


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
    p = basis_permutation(ops)
    a = _full_numpy(ops, op)
    if a.ndim == 2:
        return a[np.ix_(p, p)]
    if a.ndim == 4:
        a = a[np.ix_(p, p, p, p)]
        return a.transpose(0, 2, 1, 3).reshape(9, 9)
    raise ValueError(f"expected a one- or two-site operator, got {a.ndim} legs")
