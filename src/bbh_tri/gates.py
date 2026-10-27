"""Imaginary-time gates (SPEC §4.2): NN triangles (option A) and routed NNN bonds.

H = sum over up-triangles of h_tri; the up-triangle anchored at site b = (x, y) is
    a = b + (1, 0),  b,  c = b + (0, 1),
with bonds v = (b, a), h = (b, c), d = (a, c). Every NN bond lies in exactly one up-triangle, and
the triangle anchored at b contains the 'h', 'v' bonds of b and the 'd' bond of a.

exp(-tau h_tri) is split by two SVDs into a 3-site MPO along the square-NN path a -> b -> c, given
to YASTN as ``Gate(G=[G_a, G_b, G_c], sites=(a, b, c))``. Leg order of the tensors: (ket, bra,
virtual...); the cut legs carry signature -1 on the left factor, as in ``gates.decompose_nn_gate``.
``evolution_step_`` then truncates both path bonds (a, b) and (b, c).

On a finite OBC patch, triangles cut by the boundary are replaced by 2-site gates for the bonds
they still contain (the 'd' bond routed along ``lattice.path``).

NNN bonds (J2, K2) get one 2-site gate exp(-tau h) each, routed along a square-NN path
(``lattice.path``: 3 sites for 'nnn_d', 4 for the knight classes) with identities on the inner
sites; every bond on the path is truncated. Each bond has several paths; ``variant`` picks one
cyclically, and ``evolve`` alternates it between Trotter steps (SPEC §4.2). Paths must visit each
unit-cell tensor once, which the sqrt3 cell does not allow for NNN bonds (``lattice.gate_paths``).
"""

from yastn import eigh, exp, ncon, tensordot
from yastn.tn.fpeps import Gate
from yastn.tn.fpeps.gates import gate_nn_exp

from . import lattice, operators

SVD_TOL = 1e-14


def kron(*ops):
    """Tensor product of operators, legs (ket0, bra0, ket1, bra1, ...)."""
    out = ops[0]
    for o in ops[1:]:
        out = tensordot(out, o, axes=((), ()))
    return out


def triangle_hamiltonian(ops, h_ab, h_bc, h_ac):
    """Three-site operator h_ab ⊗ 1 + 1 ⊗ h_bc + h_ac (identity on b), legs (a, b, c) pairs."""
    I = ops.I()
    H = kron(h_ab, I) + kron(I, h_bc) + kron(h_ac, I).transpose(axes=(0, 1, 4, 5, 2, 3))
    return H + 0 * kron(I, I, I)  # all symmetry blocks present before eigh (cf. gate_nn_exp)


def exp_mpo(step, H):
    """[G_a, G_b, G_c] with G_a G_b G_c = exp(-step H) for a Hermitian three-site H."""
    D, U = eigh(H.fuse_legs(axes=((0, 2, 4), (1, 3, 5))), axes=(0, 1))
    G = ncon((U, exp(D, step=-step), U.conj()), ([-1, 1], [1, 2], [-2, 2]))
    G = G.unfuse_legs(axes=(0, 1)).transpose(axes=(0, 3, 1, 4, 2, 5))
    U0, S0, V0 = G.svd_with_truncation(axes=((0, 1), (2, 3, 4, 5)), sU=-1, tol=SVD_TOL, Vaxis=0)
    S0 = S0.sqrt()
    R = S0.broadcast(V0, axes=0)  # (v0, ket_b, bra_b, ket_c, bra_c)
    U1, S1, V1 = R.svd_with_truncation(axes=((0, 1, 2), (3, 4)), sU=-1, tol=SVD_TOL, Vaxis=0)
    S1 = S1.sqrt()
    return [S0.broadcast(U0, axes=2),
            S1.broadcast(U1, axes=3).transpose(axes=(1, 2, 0, 3)),
            S1.broadcast(V1, axes=0).transpose(axes=(1, 2, 0))]


def up_triangle(b):
    """Sites (a, b, c) of the up-triangle anchored at b, in MPO path order."""
    return lattice.shift(b, lattice.NN["v"]), b, lattice.shift(b, lattice.NN["h"])


def nn_gates(geo, model, ops, step):
    """First-order list of gates exp(-step h) covering every NN bond of ``geo`` once (NNN bonds:
    :func:`nnn_gates`; both: :func:`model_gates`).

    Infinite lattice: one triangle MPO per unit-cell site. OBC: triangle MPOs where all three
    sites are inside, 2-site gates for the remaining bonds.
    """
    hs = {c: h for c, h in model.bond_hamiltonians(ops).items() if c in lattice.NN_CLASSES}
    gates, covered = [], set()
    for b in geo.sites():
        a, _, c = up_triangle(b)
        present = {"v": (b, a), "h": (b, c), "d": (a, c)}
        present = {cls: bd for cls, bd in present.items() if all(lattice.inside(geo, s) for s in bd)}
        terms = {cls: bd for cls, bd in present.items() if cls in hs}
        if not terms:
            continue
        if len(present) == 3:
            zero = 0 * operators.h_bond(ops, 0.0, 0.0)
            H = triangle_hamiltonian(ops, hs.get("v", zero), hs.get("h", zero), hs.get("d", zero))
            gates.append(Gate(G=exp_mpo(step, H), sites=(a, b, c)))
            covered.update(terms.values())
            continue
        for cls, bd in terms.items():
            g = gate_nn_exp(step, ops.I(), hs[cls])
            sites = lattice.path(geo, bd) if cls == "d" else bd
            gates.append(g._replace(sites=tuple(sites)))
            covered.add(bd)
    if lattice.is_finite(geo):  # bonds in no up-triangle anchored inside the patch
        for cls, h in hs.items():
            for bd in lattice.bonds(geo, cls):
                if tuple(bd) not in covered:
                    g = gate_nn_exp(step, ops.I(), h)
                    sites = lattice.path(geo, bd) if cls == "d" else bd
                    gates.append(g._replace(sites=tuple(sites)))
    return gates


def nnn_gates(geo, model, ops, step, variant=0):
    """First-order list of gates exp(-step h), one per NNN bond of ``geo``, each routed along
    ``lattice.path(geo, bond, variant)``. Raises ValueError on a cell without such a path (sqrt3)."""
    out = []
    for cls, h in model.bond_hamiltonians(ops).items():
        if cls not in lattice.NNN_CLASSES:
            continue
        g = gate_nn_exp(step, ops.I(), h)
        out += [g._replace(sites=lattice.path(geo, bd, variant)) for bd in lattice.bonds(geo, cls)]
    return out


def model_gates(geo, model, ops, step, variant=0):
    """First-order list covering every coupled bond once: :func:`nn_gates` then :func:`nnn_gates`."""
    return nn_gates(geo, model, ops, step) + nnn_gates(geo, model, ops, step, variant)


def trotter2(gates_half):
    """Second-order Trotter step from gates built at step tau/2: the list followed by its reverse.

    (YASTN's ``distribute(symmetrize=True)`` does this only for gates it builds itself.)
    """
    return list(gates_half) + list(gates_half[::-1])
