"""Measurements on a PEPS with a converged CTM environment (SPEC §2.2, §4.3).

Bond energies per class use reduced density matrices where YASTN has one ('h' rdm1x2, 'v' rdm2x1,
'd' rdm2x2_anti_diagonal, 'nnn_d' rdm2x2_diagonal). Knight-move NNN classes have none; they use
``measure_nsite_exact`` on the operator expansion of h = J S·S + K (S·S)^2 into single-site
products. Never use ``measure_nn()`` without a bond or ``geometry.bonds()`` (SPEC §1).

A two-site operator O (legs out1, in1, out2, in2) is evaluated against an rdm (legs s0, s0', s1,
s1') as tensordot(O, rdm, axes=((0, 1, 2, 3), (1, 0, 3, 2))) / Tr rdm; checked against
``measure_nn`` on a complex state with a non-Hermitian operator.
"""

import numpy as np
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site
from yastn.tn.fpeps.envs.rdm import rdm1x2, rdm2x1, rdm2x2_anti_diagonal, rdm2x2_diagonal

from . import lattice, operators
from .hamiltonian import energy_per_site

RDM_CLASSES = ("h", "v", "d", "nnn_d")


def ctm_env(psi, chi, corner_tol=1e-12, max_sweeps=200, truncation_f=None, return_info=False):
    """CTM environment (method '2x2', h/v moves) iterated to corner convergence (SPEC §4.3).

    ``truncation_f``: custom projector mask (e.g. ``yastn_compat.gap_safe_truncation``);
    ``return_info``: return (env, CTMRG_out) to see whether the corners converged.
    """
    env = fpeps.EnvCTM(psi, init="eye")
    info = env.iterate_(opts_svd={"D_total": chi}, moves="hv", method="2x2", max_sweeps=max_sweeps,
                        corner_tol=corner_tol, truncation_f=truncation_f)
    return (env, info) if return_info else env


def bond_rdm(psi, env, bond, cls):
    """Two-site rdm with legs (bond[0] pair, bond[1] pair) for an rdm-supported class."""
    s0, s1 = bond
    if cls == "h":
        return rdm1x2(s0, psi, env)[0]
    if cls == "v":
        return rdm2x1(s0, psi, env)[0]
    if cls == "d":  # s0 bottom-left, s1 = s0 + (-1, 1) top-right; anchor is the upper-left site
        return rdm2x2_anti_diagonal(Site(s1[0], s0[1]), psi, env)[0]
    if cls == "nnn_d":  # s0 upper-left, s1 = s0 + (1, 1)
        return rdm2x2_diagonal(s0, psi, env)[0]
    raise ValueError(f"no rdm for class {cls!r}")


def rdm_expectation(op2, rdm):
    """<O> of a two-site operator from a two-site rdm."""
    num = yastn.tensordot(op2, rdm, axes=((0, 1, 2, 3), (1, 0, 3, 2))).to_number()
    return num / yastn.trace(rdm, axes=((0, 2), (1, 3))).to_number()


def bbh_terms(ops, J, K):
    """[(coefficient, A, B)] with J S·S + K (S·S)^2 = sum coefficient A ⊗ B (single-site A, B).

    S·S = sum_a c_a X_a ⊗ Y_a with (c, X, Y) in {(1, Sz, Sz), (1/2, S+, S-), (1/2, S-, S+)}, so
    (S·S)^2 = sum_ab c_a c_b (X_a X_b) ⊗ (Y_a Y_b).
    """
    Sz, Sp, Sm = ops.sz(), ops.sp(), ops.sm()
    ss = [(1.0, Sz, Sz), (0.5, Sp, Sm), (0.5, Sm, Sp)]
    out = [(J * c, X, Y) for c, X, Y in ss] if J != 0 else []
    if K != 0:
        out += [(K * ca * cb, Xa @ Xb, Ya @ Yb) for ca, Xa, Ya in ss for cb, Xb, Yb in ss]
    return out


def bond_energy(psi, env, ops, bond, cls, J, K):
    """<J S·S + K (S·S)^2> on one bond."""
    if cls in RDM_CLASSES:
        return rdm_expectation(operators.h_bond(ops, J, K), bond_rdm(psi, env, bond, cls))
    terms = (operators.two_site_terms(ops, operators.h_bond(ops, J, K)) if operators.is_matrix_built(ops)
             else bbh_terms(ops, J, K))  # SymSpin1: S+ has no definite charge
    return sum(c * env.measure_nsite_exact(A, B, sites=list(bond)) for c, A, B in terms)


def class_energies(psi, env, ops, model):
    """{class: mean bond energy} over the bonds of ``psi.geometry`` for every coupled class.

    A class with no bonds on a finite patch (e.g. 'nnn_k2' on 2 rows) is left out.
    """
    out = {}
    for cls, (J, K) in model.couplings().items():
        bs = lattice.bonds(psi.geometry, cls)
        if not bs:
            continue
        out[cls] = sum(bond_energy(psi, env, ops, b, cls, J, K) for b in bs) / len(bs)
    return out


def energy(psi, env, ops, model):
    """(E/site, {class: mean bond energy}); real parts.

    Each class mean is weighted by its bonds per site. On the infinite lattice that weight is 1
    (form (1) of the paper, :func:`hamiltonian.energy_per_site`); on a finite OBC patch it is
    len(bonds) / len(sites) < 1, so E/site = (sum of all bond energies) / (number of sites).
    """
    e = {c: float(np.real(v)) for c, v in class_energies(psi, env, ops, model).items()}
    n = len(psi.geometry.sites())
    w = {c: len(lattice.bonds(psi.geometry, c)) / n for c in e}
    return energy_per_site({c: w[c] * v for c, v in e.items()}), e


def site_expectations(env, ops_dict):
    """{name: {site: <op>}} for single-site operators."""
    return {k: env.measure_1site(o) for k, o in ops_dict.items()}


def order_parameters(psi, env, ops):
    """{site: (m, |Q|)}: moment |<S>| and quadrupole norm |<Q>| per site (SPEC §2.2).

    In U1 only Sz and Q2 enter; the other components vanish by symmetry.
    """
    S = site_expectations(env, operators.spin_vector(ops))
    Q = site_expectations(env, operators.quadrupole(ops))
    return {s: (operators.vector_norm({k: v[s] for k, v in S.items()}),
                operators.vector_norm({k: v[s] for k, v in Q.items()})) for s in psi.geometry.sites()}


# --- Order at ordering vectors, correlations, chirality, correlation lengths (SPEC §4.3, M6) ---
#
# Momenta q are given as (q_row, q_col) in units of 2π: site (row, col) = col·a1 + row·a2 carries
# the phase exp(2πi (q_row·row + q_col·col)), i.e. q·a2 = 2π q_row and q·a1 = 2π q_col. K is the
# 3-sublattice momentum (phase 2π·sublattice/3); M1, M2, M3 are the three M points, at which the
# NN directions a1, a2, a1 - a2 carry the signs (-, +, -), (+, -, -), (-, -, +). Expectation
# values are real, so F(-q) = F(q)*; K' = -K is folded into K (see :func:`ordered_moments`).
MOMENTA = {"Gamma": (0.0, 0.0), "K": (-1 / 3, 1 / 3), "M1": (0.0, 0.5), "M2": (0.5, 0.0), "M3": (0.5, 0.5)}
QQ_SHIFT = -8 / 3  # Q_i·Q_j = 2 (S_i·S_j)^2 + S_i·S_j - 8/3 (operators.quadrupole normalization)


def _phase(q, site):
    return np.exp(2j * np.pi * (q[0] * site[0] + q[1] * site[1]))


def commensurate(geo, q, tol=1e-10):
    """Whether the phase of ``q`` is the same on all sites sharing a unit-cell tensor of ``geo``.

    Samples rows 0..2 Nx and columns 0..2 Ny, which contains a translate of every site by each
    cell period (the periods have components at most (Nx, Ny))."""
    ref = {}
    for r in range(2 * geo.Nx + 1):
        for c in range(2 * geo.Ny + 1):
            k = geo.site2index((r, c))
            p = _phase(q, (r, c))
            if abs(ref.setdefault(k, p) - p) > tol:
                return False
    return True


def _star_weight(q):
    """Number of distinct momenta in {q, -q} modulo the reciprocal lattice (1 for Γ and M, 2 for K)."""
    return 1 if all(abs((2 * x) % 1) < 1e-12 or abs((2 * x) % 1 - 1) < 1e-12 for x in q) else 2


def ordered_moments(psi, env, ops, momenta=None):
    """{name: (m(q), Q(q))}: the spin and quadrupole order at ordering vector q from the one-site
    expectation values,

        F_X(q) = (1/N) sum_s exp(-i q·r_s) <X_s>,   X(q) = sqrt(w_q sum_components |F_X(q)|^2),

    with s over the N unit-cell sites and w_q = 2 for K (folding in K' = -K), 1 for Γ and M. So
    120° order with moment m gives m(K) = m, FM m(Γ) = m, a stripe at M_i m(M_i) = m, tetrahedral
    order m(M_i) = m/√3 each; FQ |Q|(Γ), AFQ Q(K) = |Q| (and Q(Γ) = 0). Over a complete set of
    cell momenta (sqrt3: Γ, K; (2, 2): Γ, M1-3) the squares add up to the mean of m^2 (or |Q|^2).

    ``momenta``: names in :data:`MOMENTA` or {name: q}; default all of :data:`MOMENTA` commensurate
    with the cell. An explicitly requested incommensurate q raises. Symmetric configs: only the
    charge-neutral components enter (as in :func:`order_parameters`).
    """
    geo = psi.geometry
    if momenta is None:
        qs = {k: q for k, q in MOMENTA.items() if commensurate(geo, q)}
    else:
        qs = {k: MOMENTA[k] for k in momenta} if not isinstance(momenta, dict) else dict(momenta)
        bad = [k for k, q in qs.items() if not commensurate(geo, q)]
        if bad:
            raise ValueError(f"momenta {bad} are incommensurate with the unit cell {tuple(geo.dims)}")
    S = site_expectations(env, operators.spin_vector(ops))
    Q = site_expectations(env, operators.quadrupole(ops))
    sites = list(geo.sites())

    def amp(vals, q):
        F = [sum(np.conj(_phase(q, s)) * complex(v[s]) for s in sites) / len(sites) for v in vals.values()]
        return float(np.sqrt(_star_weight(q) * sum(abs(f) ** 2 for f in F)))

    return {k: (amp(S, q), amp(Q, q)) for k, q in qs.items()}


def _pair_terms(ops, J, K):
    """[(c, A, B)] with J S·S + K (S·S)^2 = sum c A ⊗ B for single-site A, B of definite charge."""
    if operators.is_matrix_built(ops):
        M = operators.as_matrix(operators._DENSE, operators.h_bond(operators._DENSE, J, K))
        return [(c, A, B) for c, (A, B) in operators.matrix_terms(ops, M)]
    return bbh_terms(ops, J, K)


CORRELATORS = {"SS": ((1.0, 0.0), 0.0), "QQ": ((1.0, 2.0), QQ_SHIFT)}  # name: ((J, K), constant)
DIRECTIONS = {"h": (0, 1), "v": (1, 0)}  # a1 (along a row), a2 (along a column)


def correlations(psi, env, ops, dirn="h", rmax=6, kinds=("SS", "QQ"), connected=False, sites=None):
    """{kind: {s0: array}}: <X_s0 · X_{s0 + r e}> for r = 1..rmax along e = a1 ('h') or a2 ('v'),
    X = S ('SS') or the quadrupole Q ('QQ', via Q·Q = 2 (S·S)^2 + S·S - 8/3).

    Each value is an exact contraction of the width-one line between the two sites in the CTM
    environment (``EnvCTM.measure_line``); cost grows like rmax^2 per start site. ``connected``
    subtracts <X_s0>·<X_s> (charge-neutral components only in symmetric configs, where the others
    vanish). ``sites``: start sites (default: the unit-cell sites).
    """
    e = DIRECTIONS[dirn]
    sites = list(psi.geometry.sites()) if sites is None else list(sites)
    one = {"SS": operators.spin_vector(ops), "QQ": operators.quadrupole(ops)}
    out = {}
    for kind in kinds:
        (J, K), const = CORRELATORS[kind]
        terms = _pair_terms(ops, J, K)
        mean = site_expectations(env, one[kind]) if connected else None
        out[kind] = {}
        for s0 in sites:
            vals = []
            for r in range(1, rmax + 1):
                s1 = Site(s0[0] + r * e[0], s0[1] + r * e[1])
                v = sum(c * env.measure_line(A, B, sites=[s0, s1]) for c, A, B in terms) + const
                if connected:
                    v -= sum(complex(m[_cell_site(psi, s0)]) * complex(m[_cell_site(psi, s1)]) for m in mean.values())
                vals.append(float(np.real(v)))
            out[kind][s0] = np.array(vals)
    return out


def _cell_site(psi, site):
    """The unit-cell site (a key of ``measure_1site`` output) holding the tensor of ``site``."""
    k = psi.geometry.site2index(site)
    return next(s for s in psi.geometry.sites() if psi.geometry.site2index(s) == k)


TRIANGLES = {"up": ((0, 0), (0, 1), (1, 0)), "down": ((0, 1), (1, 1), (1, 0))}  # offsets, counterclockwise


def _chirality_terms(ops):
    """[(c, (A, B, C))] with S_1·(S_2 × S_3) = sum c A ⊗ B ⊗ C."""
    if operators.is_matrix_built(ops):
        return [(c, tuple(legs)) for c, legs in operators.matrix_terms(ops, operators.chirality_matrix())]
    Sz, Sp, Sm = ops.sz(), ops.sp(), ops.sm()
    out = []  # S_i·(S_j × S_k) = sum_cyclic (i/2) Sz_i (S+_j S-_k - S-_j S+_k)
    for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for c, B, C in ((0.5j, Sp, Sm), (-0.5j, Sm, Sp)):
            legs = [None] * 3
            legs[perm[0]], legs[perm[1]], legs[perm[2]] = Sz, B, C
            out.append((c, tuple(legs)))
    return out


def chiralities(psi, env, ops):
    """{'up' | 'down': {s: chi}}: scalar chirality <S_1·(S_2 × S_3)> on the triangles anchored at the
    unit-cell sites s, counterclockwise in the embedding a1 = (1, 0), a2 = (1/2, √3/2):
    up (s, s + a1, s + a2), down (s + a1, s + a1 + a2, s + a2). Both fit a 2x2 window
    (``EnvCTM.measure_2x2``). Product states give the triple product of the moments."""
    terms = _chirality_terms(ops)
    out = {}
    for name, offs in TRIANGLES.items():
        out[name] = {}
        for s in psi.geometry.sites():
            tri = [Site(s[0] + dr, s[1] + dc) for dr, dc in offs]
            v = sum(c * env.measure_2x2(*legs, sites=tri) for c, legs in terms)
            out[name][s] = float(np.real(v))
    return out


def correlation_lengths(env, dirn="h", k=3, index=0, charge=None, dtype="float64"):
    """Correlation lengths xi_j = -L / ln|lambda_j / lambda_0|, j = 1..k-1, in units of the NN
    distance along a1 ('h', L = Ny) or a2 ('v', L = Nx), from the leading eigenvalues of the
    period-L CTM transfer matrix through row (or column) ``index`` (``EnvCTM.transfer_matrix_spectrum``,
    numpy backend only). ``charge``: symmetry sector of the eigenvectors (None: zero charge); in
    symmetric configs correlations of charged operators decay with other sectors' eigenvalues.

    Every eigenvalue is normalized by the leading one of the zero-charge sector, which normalizes
    the state. Zero charge (None, 0 or a tuple of zeros): its k - 1 subleading eigenvalues give
    k - 1 lengths. Nonzero
    ``charge``: all k eigenvalues of that sector are kept (k lengths)."""
    L = env.Ny if dirn == "h" else env.Nx

    def spectrum(n, kk):
        lam = np.abs(np.asarray(env.transfer_matrix_spectrum(k=kk, n=n, dirn=dirn, i=index, dtype=dtype)))
        return np.sort(lam)[::-1]

    if charge is None or np.all(np.asarray(charge) == 0):  # explicit zero charge = the neutral sector
        lam = spectrum(None, k)
        lam0, lam = lam[0], lam[1:]
    else:
        lam0, lam = spectrum(None, 2)[0], spectrum(charge, k)
    return -L / np.log(lam / lam0)
