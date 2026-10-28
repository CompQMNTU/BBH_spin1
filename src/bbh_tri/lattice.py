"""Triangular lattice embedded in a square PEPS (SPEC §2.1).

Coordinates: YASTN ``Site(nx, ny)`` = (row, col). Primitive vectors in (row, col) components:
a1 = (0, 1), a2 = (1, 0), at 60°; the third NN bond is a1 - a2 = (-1, 1), YASTN's diagonal.
A vector (dr, dc) = dc*a1 + dr*a2 has |v|^2 = dc^2 + dr^2 + dc*dr.

Bond classes (one bond of each class per site): NN 'h', 'v', 'd'; NNN (|r| = sqrt 3) 'nnn_d',
'nnn_k1', 'nnn_k2' (k = knight move). Third neighbors (|r| = 2) are listed for future use.

Gates are routed along square-NN paths (:func:`path`) that visit each unit-cell tensor once.

Bonds are built here, not taken from ``geometry.bonds()``: that list breaks on OBC triangular
patches (``Bond(None, ...)``) and has no NNN classes. ED and PEPS must both use :func:`bonds`.
"""

from itertools import permutations

import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Bond, Site

NN = {"h": (0, 1), "v": (1, 0), "d": (-1, 1)}
NNN = {"nnn_d": (1, 1), "nnn_k1": (-1, 2), "nnn_k2": (2, -1)}
THIRD = {"3rd_h": (0, 2), "3rd_v": (2, 0), "3rd_d": (-2, 2)}
VECTORS = {**NN, **NNN, **THIRD}
NN_CLASSES = tuple(NN)
NNN_CLASSES = tuple(NNN)
# Bonds along the horizontal chains (row unchanged). All other NN and all NNN bonds connect
# different chains; the spatial anisotropy J_anis acts on those (hamiltonian.Model).
CHAIN_CLASSES = tuple(c for c in NN_CLASSES + NNN_CLASSES if VECTORS[c][0] == 0)


def norm2(v):
    """|v|^2 in units of the NN distance for v = (dr, dc) = dc*a1 + dr*a2."""
    dr, dc = v
    return dc * dc + dr * dr + dc * dr


def sublattice(site):
    """3-sublattice index (col - row) mod 3; equals TriangularLattice().site2index for the sqrt3 cell."""
    return (site[1] - site[0]) % 3


def shift(site, v):
    return Site(site[0] + v[0], site[1] + v[1])


def geometry(kind="sqrt3", dims=None):
    """Triangular geometry for a unit cell (SPEC §4.4) or a finite OBC patch.

    kind: 'sqrt3' (3 tensors, default TriangularLattice), 'patch' (infinite, full_patch, all
    dims[0]*dims[1] sites unique), or 'obc' (finite patch; always full_patch=True, since without
    it YASTN ignores dims).
    """
    if kind == "sqrt3":
        return fpeps.TriangularLattice()
    if kind == "patch":
        return fpeps.TriangularLattice(dims=tuple(dims), boundary="infinite", full_patch=True)
    if kind == "obc":
        return fpeps.TriangularLattice(dims=tuple(dims), boundary="obc", full_patch=True)
    raise ValueError(f"unknown geometry kind {kind!r}")


def same_geometry(a, b):
    """Whether two TriangularLattice geometries are the same cell (dims, boundary, full_patch)."""
    return (type(a), tuple(a.dims), a.boundary, a.full_patch) == (type(b), tuple(b.dims), b.boundary, b.full_patch)


def is_finite(geo):
    return geo.boundary == "obc"


def inside(geo, site):
    """Whether a site belongs to the lattice (always true on an infinite lattice)."""
    if not is_finite(geo):
        return True
    return 0 <= site[0] < geo.Nx and 0 <= site[1] < geo.Ny


def bonds(geo, cls):
    """All bonds of class ``cls``: (s, s + v) for every site s of the geometry.

    Infinite lattice: s runs over the unit-cell sites, so each translation class of bonds appears
    exactly once (one bond per site and class). Finite OBC patch: s runs over the patch and bonds
    leaving it are dropped.
    """
    v = VECTORS[cls]
    out = []
    for s in geo.sites():
        t = shift(s, v)
        if inside(geo, t):
            out.append(Bond(s, t))
    return tuple(out)


def paths(geo, bond):
    """All shortest square-NN paths from bond[0] to bond[1] that stay inside the lattice.

    Each path is a tuple of Sites, starting at bond[0]; consecutive sites differ by one row or one
    column step. Order is deterministic: sorted by the step sequence, so variants can be alternated
    between sweeps (SPEC §4.2). For NN h/v bonds the only path is the bond itself.
    """
    dr, dc = bond[1][0] - bond[0][0], bond[1][1] - bond[0][1]
    steps = [(1 if dr > 0 else -1, 0)] * abs(dr) + [(0, 1 if dc > 0 else -1)] * abs(dc)
    out = []
    for seq in sorted(set(permutations(steps))):
        p = [bond[0]]
        for st in seq:
            p.append(shift(p[-1], st))
        if all(inside(geo, s) for s in p):
            out.append(tuple(p))
    return out


def gate_paths(geo, bond):
    """The paths of :func:`paths` a gate can be routed along: those that visit every unit-cell
    tensor at most once. ``evolution_step_`` applies a gate on a path through one tensor twice
    without an error, and the result is wrong (docs/yastn_findings.md). On the sqrt3 cell every
    NNN pair shares a tensor, so NNN gates need a patch cell, e.g. (3, 3) or (6, 6); on (2, 2)
    one path per knight class remains."""
    return [p for p in paths(geo, bond) if len({geo.site2index(s) for s in p}) == len(p)]


def path(geo, bond, variant=0):
    """One path of :func:`gate_paths`, chosen cyclically by ``variant``."""
    ps = gate_paths(geo, bond)
    if not ps:
        raise ValueError(f"no square-NN path for {bond} inside the lattice that visits each unit-cell tensor "
                         "once; NNN gates need a patch cell such as geometry('patch', (3, 3)) (sqrt3: every "
                         "NNN pair shares a tensor)")
    return ps[variant % len(ps)]
