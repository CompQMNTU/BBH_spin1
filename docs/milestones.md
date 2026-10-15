# Milestone notes

## M0 — Recon (done, PRs #3, #4)
See `docs/yastn_findings.md`. U1/U2 resolved, U3 partly (2×3 window runs; cost at target D open), U4 resolved with `bbh_tri.yastn_compat.exact_svd_backward`.

## M1 — Operators & lattice (2026-10-15)
Modules:
- `lattice.py`: bond vectors per class (`h`, `v`, `d`, `nnn_d`, `nnn_k1`, `nnn_k2`, plus third neighbors), |v|², sublattice map, geometry builder (`sqrt3`, infinite `patch`, finite `obc` — always `full_patch=True`), bond lists, square-NN paths.
- `operators.py`: S·S, (S·S)², h(J, K), quadrupole Q, spin vector, matrices in the canonical basis Sz = (+1, 0, −1).
- `hamiltonian.py`: `Model(theta, J2, K2, J_anis)` → couplings and bond Hamiltonians per class; E/site = sum of class means. J_anis (paper Sec. IV F) scales every bond between horizontal chains: `v`, `d` and, by our choice, all NNN classes; classes with zero coupling are dropped.

Design decisions:
- **Bonds are built by `lattice.bonds`, never `geometry.bonds()`.** Infinite lattice: one bond per unit-cell site and class, starting at the cell sites (`h`, `v` identical to YASTN's; `d` is YASTN's diagonal translated by (1, 0)). OBC: bonds leaving the patch are dropped, so no `Bond(None, …)`. ED and PEPS must both use this function (SPEC §3).
- **Paths:** `lattice.paths` returns all shortest square-NN paths in a fixed order (2 for `d` and `nnn_d`, 3 for the knight moves); `lattice.path(geo, bond, variant)` cycles through them so sweeps can alternate (SPEC §4.2). The 2×3 paths are exactly those verified against exact vectors in `tests/test_routing.py`.
- **U1:** built from Sz, S± (YASTN's U1 `Spin1` has no Sx, Sy). `quadrupole` returns only Q2 = (3Sz² − 2)/√3 and `spin_vector` only Sz for U1; the other components change Sz and vanish in U1 states (SPEC §2.2).
- **Basis:** U1 `to_numpy` orders Sz = (−1, 0, +1); `operators.as_matrix` permutes to (+1, 0, −1) so dense and U1 compare directly.

Tests: `tests/test_lattice.py`, `tests/test_operators.py` (dense and U1), `tests/test_hamiltonian.py`.

Open:
- `tests/yastn_utils.py` (M0 helpers) still has its own `diag_path` / `ss_yastn`; switch the M0 tests to `bbh_tri` when they are next touched.

## M2 — D = 1 product states (2026-10-15)
Modules:
- `states.py`: phase seeds (`FM`, `FQ`, `AFQ` with real |x>, |y>, |z> directors, `SU3` triad), mean-field E/site of a 3-sublattice product state (NN classes couple AB, BC, CA; NNN classes AA, BB, CC; all couplings incl. J_anis), `optimize_product_state` (L-BFGS over complex single-site states from the seeds plus random starts), `moment`, `product_peps` (vectors in the canonical basis; complex vectors need a complex config; U1 needs Sz eigenstates).
- `measure.py`: `ctm_env`, bond energy per class from rdms (`h` rdm1x2, `v` rdm2x1, `d` rdm2x2_anti_diagonal, `nnn_d` rdm2x2_diagonal) or, for the knight moves, `measure_nsite_exact` on the 12-term single-site expansion of J S·S + K (S·S)²; `energy` (E/site, per class); `order_parameters` (m, |Q| per site; U1 uses Sz, Q2 only).

Findings:
- rdm contraction: `tensordot(O, rdm, axes=((0,1,2,3),(1,0,3,2)))` / Tr; the other pairing fails the signature check. Verified against `measure_nn` with a non-Hermitian operator on a complex state (a real symmetric h cannot tell the orders apart).
- `rdm2x2_anti_diagonal(anchor)` returns legs (bottom-left, top-right) = our `d` bond order with anchor (x−1, y).
- `measure_nsite_exact` and `measure_2x2` accept U1 charged operators (S±).
- A class with no bonds on a finite patch (`nnn_k2` on 2 rows) is left out of `class_energies`.

Acceptance (SPEC §6 M2), `tests/test_product_states.py`:
- §3 table to 1e-10 (minimizer: ~1e-15); seed formulas E_FM, E_FQ, E_AFQ over θ; 120° coherent state = 2.254 at π/4.
- FQ optimal with E = 6 sin θ, m = 0 for 3π/2 < θ ≤ θ_c^MF; above it m grows continuously from 0 and E_opt − E_FQ scales as (θ − θ_c)² (ratio 4 within 20%); reduced AFM3 moments 0.47 / 0.78 / 0.95 / 0.99 at 1.66 / 1.7 / 1.8 / 1.9 π, E(1.7π) = −4.9576.
- Full pipeline (product PEPS → CTM χ = 1 → rdm / nsite) equals the mean-field energy to 1e-10 for every coupled class, including NNN and J_anis; U1 = dense for Sz-eigenstate seeds; |Q| through CTM; OBC patches.

Open:
- SPEC §5 lists an `m=2/3` product seed for `states.py`; its definition is not in SPEC, so it is not implemented.
