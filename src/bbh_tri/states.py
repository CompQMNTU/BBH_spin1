"""Product (D = 1) states: seeds, mean-field energy and the general 3-sublattice minimizer (SPEC §3).

Single-site states are numpy 3-vectors in the canonical basis Sz = (+1, 0, -1)
(``operators.as_matrix`` convention); sublattice k = lattice.sublattice(site) = (col - row) mod 3.

On the sqrt3 x sqrt3 cell every NN class connects the three sublattice pairs (AB, BC, CA) once
and every NNN class connects each sublattice to itself, so for a product state
    E/site = sum_NN-classes a_c <h_c>_{AB,BC,CA} / 3 + sum_NNN-classes a_c <h_c>_{AA,BB,CC} / 3.
"""

import numpy as np
import scipy.optimize
import yastn
import yastn.tn.fpeps as fpeps

from . import lattice, operators

SQRT2 = np.sqrt(2)


def basis_state(sz):
    """|Sz = sz> in the canonical basis (+1, 0, -1)."""
    v = np.zeros(3)
    v[(1, 0, -1).index(sz)] = 1.0
    return v


def seed_vectors(kind):
    """Sublattice states (A, B, C) for the phase seeds of SPEC §3 / §4.5.

    'FM' all |+1> (moment along z); 'FQ' all |0> (director along z); 'AFQ' directors x, y, z,
    i.e. the time-reversal-invariant states |x> ~ |-1> - |+1>, |y> ~ |-1> + |+1> (real after a
    global phase), |z> = |0>; 'SU3' the orthogonal triad |+1>, |0>, |-1>.
    With J = cos(theta), K = sin(theta): E_SU3 = -J + 4K and E_AFQ = 3K, equal wherever J = K
    (theta = pi/4 and 5pi/4). AFM3 has no fixed seed: use :func:`optimize_product_state`.
    """
    up, zero, down = basis_state(1), basis_state(0), basis_state(-1)
    if kind == "FM":
        return (up, up, up)
    if kind == "FQ":
        return (zero, zero, zero)
    if kind == "AFQ":
        return ((down - up) / SQRT2, (down + up) / SQRT2, zero)
    if kind == "SU3":
        return (up, zero, down)
    raise ValueError(f"unknown seed {kind!r}")


def _bond_terms(model, ops):
    """[(9x9 bond Hamiltonian in the canonical basis, sublattice pairs)] for the classes of ``model``."""
    nn, nnn = [(0, 1), (1, 2), (2, 0)], [(0, 0), (1, 1), (2, 2)]
    return [(operators.as_matrix(ops, h), nn if c in lattice.NN_CLASSES else nnn)
            for c, h in model.bond_hamiltonians(ops).items()]


def _energy(terms, vs):
    return sum(np.real(np.vdot(np.kron(vs[a], vs[b]), H @ np.kron(vs[a], vs[b]))) for H, pairs in terms for a, b in pairs) / 3


def product_energy(model, vectors, ops=None):
    """Mean-field E/site of the 3-sublattice product state ``vectors`` = (A, B, C)."""
    ops = operators.spin1("dense") if ops is None else ops
    vs = [np.asarray(v, dtype=complex) / np.linalg.norm(v) for v in vectors]
    return float(_energy(_bond_terms(model, ops), vs))


def _to_vectors(x):
    z = x[:9] + 1j * x[9:]
    return [z[3 * k:3 * k + 3] / np.linalg.norm(z[3 * k:3 * k + 3]) for k in range(3)]


def optimize_product_state(model, restarts=20, seed=0):
    """Lowest-energy general 3-sublattice product state (complex single-site states).

    Minimizes :func:`product_energy` with L-BFGS from the seeds of :func:`seed_vectors` plus
    ``restarts`` random starts. Returns (energy, (A, B, C)). This is the AFM3 seed of SPEC §3: the
    optimum is a reduced-moment coplanar state, not the 120° coherent state.
    """
    terms = _bond_terms(model, operators.spin1("dense"))
    rng = np.random.default_rng(seed)
    starts = [np.concatenate([np.concatenate(v), np.zeros(9)]) for v in map(seed_vectors, ("FM", "FQ", "AFQ", "SU3"))]
    starts += [rng.normal(size=18) for _ in range(restarts)]
    best = None
    for x0 in starts:
        res = scipy.optimize.minimize(lambda x: _energy(terms, _to_vectors(x)), x0, method="L-BFGS-B",
                                      options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.fun), tuple(_to_vectors(best.x))


def moment(v, ops=None):
    """|<S>| of a single-site state."""
    ops = operators.spin1("dense") if ops is None else ops
    v = np.asarray(v, dtype=complex) / np.linalg.norm(v)
    return operators.vector_norm({k: np.vdot(v, operators.as_matrix(ops, s) @ v)
                                  for k, s in operators.spin_vector(ops).items()})


def product_peps(geo, vectors, ops):
    """D = 1 PEPS on ``geo`` with site state vectors[lattice.sublattice(site)].

    Vectors are in the canonical basis. Complex vectors need a complex config
    (``operators.spin1(..., default_dtype='complex128')``); with a symmetry each vector must be a
    charge eigenstate (in U1xU1 and Z2xZ2 its phase is dropped).
    """
    tensors = {}
    for s in geo.sites():
        v = np.asarray(vectors[lattice.sublattice(s)])
        if np.iscomplexobj(v) and np.any(v.imag != 0) and operators.is_dense(ops) \
                and "complex" not in str(ops.config.default_dtype):
            raise ValueError("complex state vector needs a complex config")
        tensors[s] = operators.state_vector(ops, v)
    return fpeps.product_peps(geo, tensors)


def random_product_peps(geo, ops, seed=0):
    """D = 1 PEPS with an independent random real state on every site of ``geo`` (dense only)."""
    if not operators.is_dense(ops):
        raise ValueError("random product states need a dense config")
    rng = np.random.default_rng(seed)
    basis = [ops.vec_z(val=sz) for sz in (1, 0, -1)]
    tensors = {}
    for s in geo.sites():
        c = rng.normal(size=3)
        tensors[s] = c[0].item() * basis[0] + c[1].item() * basis[1] + c[2].item() * basis[2]
    return fpeps.product_peps(geo, tensors)


def to_dense(psi, ops, dense_ops=None):
    """Dense copy of a symmetric PEPS (``ops`` its operator set); the state is unchanged. Used to
    cross-check symmetric runs.

    Site tensors have legs (t, l, b, r, p) with p = fused (physical, ancilla), the ancilla being
    one-dimensional and carrying the charge of a product-state seed. Each virtual leg is expanded
    over the union of the sectors on both ends of its bond (a tensor lists only the sectors of its
    own blocks), the physical leg over the full local space and then rotated to the canonical basis.
    """
    dense_ops = operators.spin1("dense") if dense_ops is None else dense_ops
    B = operators.basis_matrix(ops)
    other = {0: ("t", 2), 1: ("l", 3), 2: ("b", 0), 3: ("r", 1)}
    phys = ops.space()
    tensors = {}
    for s in psi.sites():
        t = psi[s].unfuse_legs(axes=4)  # (t, l, b, r, physical, ancilla)
        legs = {4: phys if t.s[4] == phys.s else phys.conj()}
        for ax, (d, oax) in other.items():
            nb = psi.nn_site(s, d)
            own = t.get_legs(axes=ax)
            legs[ax] = own if nb is None else yastn.legs_union(own, psi[nb].get_legs(axes=oax).conj())
        a = np.moveaxis(np.tensordot(t.to_numpy(legs=legs), B, axes=((4,), (1,))), -1, 4)  # physical -> canonical
        if np.abs(a.imag).max() <= 1e-14 * np.abs(a).max():
            a = a.real
        T = yastn.Tensor(config=dense_ops.config, s=t.s, dtype="complex128" if np.iscomplexobj(a) else "float64")
        T.set_block(val=a, Ds=a.shape)
        tensors[s] = T.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
    return fpeps.Peps(psi.geometry, tensors=tensors)
