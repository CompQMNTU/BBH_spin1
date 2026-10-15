# Milestone notes

## M0 — Recon (done, PRs #3, #4)
See `docs/yastn_findings.md`. U1/U2 resolved, U3 partly (2×3 window runs; cost at target D open), U4 resolved with `bbh_tri.yastn_compat.exact_svd_backward`.

## M1 — Operators & lattice (2026-10-15)
Modules:
- `lattice.py`: bond vectors per class (`h`, `v`, `d`, `nnn_d`, `nnn_k1`, `nnn_k2`, plus third neighbors), |v|², sublattice map, geometry builder (`sqrt3`, infinite `patch`, finite `obc` — always `full_patch=True`), bond lists, square-NN paths.
- `operators.py`: S·S, (S·S)², h(J, K), quadrupole Q, spin vector, matrices in the canonical basis Sz = (+1, 0, −1).
- `hamiltonian.py`: `Model(theta, J2, K2)` → couplings and bond Hamiltonians per class; E/site = sum of class means.

Design decisions:
- **Bonds are built by `lattice.bonds`, never `geometry.bonds()`.** Infinite lattice: one bond per unit-cell site and class, starting at the cell sites (`h`, `v` identical to YASTN's; `d` is YASTN's diagonal translated by (1, 0)). OBC: bonds leaving the patch are dropped, so no `Bond(None, …)`. ED and PEPS must both use this function (SPEC §3).
- **Paths:** `lattice.paths` returns all shortest square-NN paths in a fixed order (2 for `d` and `nnn_d`, 3 for the knight moves); `lattice.path(geo, bond, variant)` cycles through them so sweeps can alternate (SPEC §4.2). The 2×3 paths are exactly those verified against exact vectors in `tests/test_routing.py`.
- **U1:** built from Sz, S± (YASTN's U1 `Spin1` has no Sx, Sy). `quadrupole` returns only Q2 = (3Sz² − 2)/√3 and `spin_vector` only Sz for U1; the other components change Sz and vanish in U1 states (SPEC §2.2).
- **Basis:** U1 `to_numpy` orders Sz = (−1, 0, +1); `operators.as_matrix` permutes to (+1, 0, −1) so dense and U1 compare directly.

Tests: `tests/test_lattice.py`, `tests/test_operators.py` (dense and U1), `tests/test_hamiltonian.py`.

Open:
- **J_anis** (SPEC §0, "the anisotropy knob J_anis of the paper") is not implemented: its definition is not in SPEC. Needs the paper's equation before adding it to `Model`.
- `tests/yastn_utils.py` (M0 helpers) still has its own `diag_path` / `ss_yastn`; switch the M0 tests to `bbh_tri` when they are next touched.
