"""M1: lattice geometry, bond classes and gate paths (SPEC §2.1)."""

from math import comb

import pytest
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site

from bbh_tri import lattice

ALL_CLASSES = lattice.NN_CLASSES + lattice.NNN_CLASSES


@pytest.mark.parametrize("vectors,expected", [(lattice.NN, 1), (lattice.NNN, 3), (lattice.THIRD, 4)])
def test_vector_norms(vectors, expected):
    """|v|^2 = n1^2 + n2^2 + n1 n2 gives 1, 3, 4 for the NN, NNN and third shells (SPEC §2.1)."""
    assert all(lattice.norm2(v) == expected for v in vectors.values())


def test_shells_are_complete():
    """Every lattice vector with |v|^2 in {1, 3, 4} is one of the listed vectors up to sign."""
    listed = {v for v in lattice.VECTORS.values()} | {(-a, -b) for a, b in lattice.VECTORS.values()}
    found = {(a, b) for a in range(-3, 4) for b in range(-3, 4) if lattice.norm2((a, b)) in (1, 3, 4)}
    assert found == listed


def test_sublattice_matches_sqrt3_cell():
    geo = fpeps.TriangularLattice()
    for r in range(-4, 5):
        for c in range(-4, 5):
            assert lattice.sublattice(Site(r, c)) == geo.site2index(Site(r, c))


def test_nnn_preserves_sublattice_nn_changes_it():
    """NNN bonds connect the same 3-sublattice, so the sqrt3 cell can hold states with J2 (SPEC §2.1);
    NNN gates cannot be routed there (test_gate_paths_visit_each_tensor_once)."""
    for r in range(-3, 4):
        for c in range(-3, 4):
            s = Site(r, c)
            assert all(lattice.sublattice(lattice.shift(s, v)) == lattice.sublattice(s) for v in lattice.NNN.values())
            assert all(lattice.sublattice(lattice.shift(s, v)) != lattice.sublattice(s) for v in lattice.NN.values())


@pytest.mark.parametrize("kind,dims,n_sites", [("sqrt3", None, 3), ("patch", (2, 2), 4), ("patch", (6, 6), 36)])
def test_infinite_bonds_one_per_site_and_class(kind, dims, n_sites):
    geo = lattice.geometry(kind, dims)
    assert len(geo.sites()) == n_sites
    for cls in ALL_CLASSES:
        bs = lattice.bonds(geo, cls)
        assert [b[0] for b in bs] == list(geo.sites())
        assert all((b[1][0] - b[0][0], b[1][1] - b[0][1]) == lattice.VECTORS[cls] for b in bs)


def test_hv_bonds_match_yastn_and_diagonal_is_a_translate():
    geo = lattice.geometry("sqrt3")
    for cls in ("h", "v"):
        assert lattice.bonds(geo, cls) == tuple(geo.bonds(cls))
    # yastn's diagonal ((x+1,y),(x,y+1)) is ours ((x,y),(x-1,y+1)) translated by (1,0): same classes
    ours = {(lattice.sublattice(b[0]), lattice.sublattice(b[1])) for b in lattice.bonds(geo, "d")}
    theirs = {(lattice.sublattice(b[0]), lattice.sublattice(b[1])) for b in geo.bonds("d")}
    assert ours == theirs


def test_obc_3x3_bond_counts():
    """OBC 3x3 ED cluster (SPEC §3): 6 h + 6 v + 4 d = 16 NN bonds; NNN 4 + 2 + 2; no None sites."""
    geo = lattice.geometry("obc", (3, 3))
    counts = {cls: len(lattice.bonds(geo, cls)) for cls in ALL_CLASSES}
    assert counts == {"h": 6, "v": 6, "d": 4, "nnn_d": 4, "nnn_k1": 2, "nnn_k2": 2}
    assert all(lattice.inside(geo, s) for cls in ALL_CLASSES for b in lattice.bonds(geo, cls) for s in b)


@pytest.mark.parametrize("kind,dims", [("sqrt3", None), ("patch", (2, 2)), ("obc", (3, 3)), ("obc", (2, 3))])
def test_paths_are_valid(kind, dims):
    geo = lattice.geometry(kind, dims)
    for cls in ALL_CLASSES:
        dr, dc = lattice.VECTORS[cls]
        for b in lattice.bonds(geo, cls):
            ps = lattice.paths(geo, b)
            assert ps, f"no path for {b}"
            # all monotone paths; inside a rectangular patch none of them can leave it
            assert len(ps) == comb(abs(dr) + abs(dc), abs(dr))
            for p in ps:
                assert p[0] == b[0] and p[-1] == b[1] and len(p) == abs(dr) + abs(dc) + 1
                assert all(abs(p[i + 1][0] - p[i][0]) + abs(p[i + 1][1] - p[i][1]) == 1 for i in range(len(p) - 1))
                assert all(lattice.inside(geo, s) for s in p)


def test_paths_match_routing_test_patch():
    """The 2x3 OBC paths are exactly those checked against ED in tests/test_routing.py (U1)."""
    geo = lattice.geometry("obc", (2, 3))
    S = Site
    assert set(lattice.paths(geo, (S(1, 0), S(0, 1)))) == {(S(1, 0), S(0, 0), S(0, 1)), (S(1, 0), S(1, 1), S(0, 1))}
    assert set(lattice.paths(geo, (S(1, 0), S(0, 2)))) == {
        (S(1, 0), S(0, 0), S(0, 1), S(0, 2)), (S(1, 0), S(1, 1), S(1, 2), S(0, 2)), (S(1, 0), S(1, 1), S(0, 1), S(0, 2))}


def test_path_variant_cycles():
    geo = lattice.geometry("patch", (3, 3))
    b = lattice.bonds(geo, "nnn_k1")[0]
    ps = lattice.paths(geo, b)
    assert len(ps) == 3 and lattice.gate_paths(geo, b) == ps
    assert [lattice.path(geo, b, i) for i in range(2 * len(ps))] == ps + ps


@pytest.mark.parametrize("kind,dims,n_paths", [
    ("sqrt3", None, {"nnn_d": 0, "nnn_k1": 0, "nnn_k2": 0}),  # every NNN pair shares a tensor
    ("patch", (2, 2), {"nnn_d": 2, "nnn_k1": 1, "nnn_k2": 1}),
    ("patch", (3, 3), {"nnn_d": 2, "nnn_k1": 3, "nnn_k2": 3}),
    ("patch", (6, 6), {"nnn_d": 2, "nnn_k1": 3, "nnn_k2": 3}),
    ("obc", (3, 3), None)])
def test_gate_paths_visit_each_tensor_once(kind, dims, n_paths):
    """A gate path may not pass a unit-cell tensor twice (yastn applies it silently and wrongly)."""
    geo = lattice.geometry(kind, dims)
    for cls in ALL_CLASSES:
        for b in lattice.bonds(geo, cls):
            gp = lattice.gate_paths(geo, b)
            assert all(len({geo.site2index(s) for s in p}) == len(p) for p in gp)
            assert set(gp) <= set(lattice.paths(geo, b))
            if cls in lattice.NN_CLASSES or n_paths is None:
                assert gp == lattice.paths(geo, b)
            else:
                assert len(gp) == n_paths[cls]
    if kind == "sqrt3":
        with pytest.raises(ValueError, match="patch cell"):
            lattice.path(geo, lattice.bonds(geo, "nnn_d")[0])
