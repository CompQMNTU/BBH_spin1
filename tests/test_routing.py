"""U1 (SPEC §1): gates on non-square-NN bonds act on the correct sites.

A 2-site gate exp(-tau S·S) given with `sites` = a square-NN path must equal the exact
2-site operator on the path's end sites, with identity on the middle sites. Checked
against exact state vectors on a 2x3 OBC patch, without truncation.
"""

import numpy as np
import pytest
import scipy.linalg as sl
import yastn
import yastn.tn.fpeps as fpeps
from yastn.tn.fpeps import Site as S

from yastn_utils import apply_2site, peps_vector, spin1, ss_numpy, ss_yastn

NX, NY = 2, 3
TAU = 0.3

# (name, square-NN path); the gate acts between path[0] and path[-1]
PATHS = [
    ("diag via (0,0)", [S(1, 0), S(0, 0), S(0, 1)]),
    ("diag via (1,1)", [S(1, 0), S(1, 1), S(0, 1)]),
    ("nnn (1,1)", [S(0, 0), S(0, 1), S(1, 1)]),
    ("knight (-1,2) A", [S(1, 0), S(0, 0), S(0, 1), S(0, 2)]),
    ("knight (-1,2) B", [S(1, 0), S(1, 1), S(1, 2), S(0, 2)]),
    ("knight (-1,2) C", [S(1, 0), S(1, 1), S(0, 1), S(0, 2)]),
]
ENVS = {"BP": fpeps.EnvBP, "NTU": fpeps.EnvNTU}


@pytest.fixture(scope="module")
def ops():
    return spin1("dense")


@pytest.fixture(scope="module")
def geometry():
    # Only h/v bonds and explicit paths are used here: the OBC diagonal bond list is broken (SPEC §1).
    return fpeps.TriangularLattice(dims=(NX, NY), boundary="obc", full_patch=True)


def entangled_state(ops, geo):
    """Product |+1>,|0>,|-1> pattern, then random symmetric h/v gates truncated to D=2.

    The reference vector is read *after* this preparation, so truncation here is harmless;
    it keeps the untruncated path gate cheap (NTU on a D~81 bond takes minutes).
    """
    seeds = {S(nx, ny): [1, 0, -1][(ny - nx) % 3] for nx in range(NX) for ny in range(NY)}
    psi = fpeps.product_peps(geo, {s: ops.vec_z(val=v) for s, v in seeds.items()})
    rng = np.random.default_rng(1)
    env = fpeps.EnvNTU(psi)
    for b in list(geo.bonds("h")) + list(geo.bonds("v")):
        A = rng.normal(size=(9, 9))
        Ht = yastn.Tensor(config=ops.config, s=(1, -1, 1, -1))
        Ht.set_block(val=(A + A.T).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3), Ds=(3, 3, 3, 3))
        g = fpeps.gates.gate_nn_exp(0.2, ops.I(), Ht)
        fpeps.evolution_step_(env, [g._replace(sites=b)], opts_svd={"D_total": 2})
    return psi


def normalized(v):
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("method", ["mpo", "nn"])
@pytest.mark.parametrize("env_name", list(ENVS))
@pytest.mark.parametrize("name,path", PATHS, ids=[p[0] for p in PATHS])
def test_path_gate_matches_exact(ops, geometry, name, path, env_name, method):
    psi = entangled_state(ops, geometry)
    v0, order = peps_vector(psi, NX, NY)

    gate = fpeps.gates.gate_nn_exp(TAU, ops.I(), ss_yastn(ops))
    env = ENVS[env_name](psi)
    fpeps.evolution_step_(env, [gate._replace(sites=tuple(path))],
                          opts_svd={"D_total": 256, "tol": 1e-15}, method=method)
    v1, _ = peps_vector(psi, NX, NY)

    U = sl.expm(-TAU * ss_numpy(ops))
    exact = apply_2site(v0, order, U, path[0], path[-1])
    wrong = apply_2site(v0, order, U, path[0], path[1])  # gate on the first path bond instead

    assert abs(np.vdot(normalized(v1), normalized(exact))) == pytest.approx(1.0, abs=1e-10)
    assert abs(np.vdot(normalized(v1), normalized(wrong))) < 0.99  # the test can tell placements apart


def test_product_state_middle_site_untouched(ops, geometry):
    """|+1>-|0>-|-1> along the diagonal path: the middle site stays |0>, ends match exact."""
    psi = fpeps.product_peps(geometry, {**{s: ops.vec_z(val=0) for s in geometry.sites()},
                                        S(1, 0): ops.vec_z(val=1), S(0, 1): ops.vec_z(val=-1)})
    v0, order = peps_vector(psi, NX, NY)
    gate = fpeps.gates.gate_nn_exp(TAU, ops.I(), ss_yastn(ops))
    fpeps.evolution_step_(fpeps.EnvBP(psi), [gate._replace(sites=(S(1, 0), S(0, 0), S(0, 1)))],
                          opts_svd={"D_total": 16, "tol": 1e-15})
    v1, _ = peps_vector(psi, NX, NY)
    exact = normalized(apply_2site(v0, order, sl.expm(-TAU * ss_numpy(ops)), S(1, 0), S(0, 1)))
    v1 = normalized(v1)
    v1 = v1 * np.sign(np.vdot(v1, exact).real)
    assert np.abs(v1 - exact).max() < 1e-12
    i_mid = order.index(S(0, 0))
    sz0 = ops.vec_z(val=0).to_numpy().argmax()  # basis index of |0>
    off = np.delete(v1, sz0, axis=i_mid)
    assert np.linalg.norm(off) < 1e-12
