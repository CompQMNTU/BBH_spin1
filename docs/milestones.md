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
- `measure.energy` weights each class mean by bonds per site (1 on the infinite lattice; len(bonds)/len(sites) on an OBC patch), so finite-patch E/site = total bond energy / sites.

Acceptance (SPEC §6 M2), `tests/test_product_states.py`:
- §3 table to 1e-10 (minimizer: ~1e-15); seed formulas E_FM, E_FQ, E_AFQ over θ; 120° coherent state = 2.254 at π/4.
- FQ optimal with E = 6 sin θ, m = 0 for 3π/2 < θ ≤ θ_c^MF; above it m grows continuously from 0 and E_opt − E_FQ scales as (θ − θ_c)² (ratio 4 within 20%); reduced AFM3 moments 0.47 / 0.78 / 0.95 / 0.99 at 1.66 / 1.7 / 1.8 / 1.9 π, E(1.7π) = −4.9576.
- Full pipeline (product PEPS → CTM χ = 1 → rdm / nsite) equals the mean-field energy to 1e-10 for every coupled class, including NNN and J_anis; U1 = dense for Sz-eigenstate seeds; |Q| through CTM; OBC patches.

Open:
- SPEC §5 lists an `m=2/3` product seed for `states.py`; its definition is not in SPEC, so it is not implemented.

## M3 — NN imaginary-time evolution (2026-10-16)
Modules:
- `gates.py`: up-triangle anchored at b = (x, y): a = b + (1, 0), c = b + (0, 1), bonds v = (b, a), h = (b, c), d = (a, c); every NN bond lies in exactly one up-triangle. exp(−τ h▲) is cut by two SVDs into a 3-site MPO along the square-NN path a → b → c and passed as `Gate(G=[G_a, G_b, G_c], sites=(a, b, c))` (legs ket, bra, virtual; left cut leg signature −1, as in `decompose_nn_gate`); `evolution_step_` then truncates both path bonds. On OBC patches, triangles cut by the boundary fall back to 2-site gates (`d` routed with `lattice.path`). `trotter2` = gates at τ/2 plus the reversed list. NNN gates: M5 (`nn_gates` raises for J₂, K₂ ≠ 0).
- `evolve.py`: second-order Trotter, τ = 0.1 → 0.05 → 0.02 → 0.01, truncation to D with EnvBP (default) or EnvNTU; each stage ends when the CTM energy (χ = D², fresh environment) changes by < 1e-7 between checks 20 steps apart, or after 400 steps. Site tensors are renormalized after every step.
- `states.random_product_peps`: independent random real state per site (dense).
- `scripts/m3_theta_scan.py` (resumable JSONL scan, state saving, warm start), `scripts/m3_plot_scan.py`.

Findings:
- **`evolution_step_` does not normalize the PEPS.** Tensor norms change by ~exp(−τE) per step and overflow/underflow to NaN after ~10³ steps (52 of 400 scan runs died with `array must not contain infs or NaNs` before the fix; θ = 1.5π grows ~1e5 per 20 steps). Once NaNs appear, `truncate_bipartite_` also raises `UnboundLocalError: pinv_c` (`_evolution.py:704`). Fix: `evolve.normalize_` after each step.
- **Unstable product seeds do not stay put.** The FM product state is an exact eigenstate of every gate, yet outside the FM phase round-off is amplified and the FM-seeded run reaches the same state as the others (often within 1e-8, so `best_per_theta` reports it under the `FM` label).
- **FM multiplet degeneracy:** from a random start in the FM phase, D = 2 converges to the exact FM energy with m = 0.74 (a superposition of FM directions). Seeded runs keep m = 1; the scan resolves ties in favor of the seeds.
- **χ = D² is not converged at the 1e-5 level.** At θ = π/2 every bond obeys (S·S)² ≥ 1, so E ≥ 3; the D = 4 scan state gives E = 2.9999942 at χ = 16 (h, v bonds 0.9999956 < 1), 3.0000107 at χ = 24 and 3.0000115 at χ = 36. Scan energies therefore carry ~2e-5 CTM error; enough for the phase diagram, but precise energies need χ ≥ 1.5 D² and the χ, 1.5χ check of SPEC §4.3.
- NTU (`which='NN'`, the cheapest) at D = 4: ~20 s per Trotter step vs ~0.3 s for BP, half in the bond metric (`_g_NN`), half in the truncation optimization; `initialization='SVD'` or smaller `max_iter` does not help. On the shared machine (load ~40) a step takes ~76 s. NTU warm start from the 80 BP winners (τ = 0.02, 0.01, `results/m3_scan_D4_NTU.jsonl`) is still running; the first finished point (θ = 0.525π, AFQ) agrees with BP to 8e-9. The NTU comparison follows in a separate commit.

Acceptance (SPEC §6 M3):
- (a) FM region: E = 3(cos θ + sin θ) to 1e-6 at θ = 0.6π, π, 1.2π from random starts at D = 2 (`tests/test_evolve.py`); in the D = 4 scan to ≲1e-9 over 0.525π–1.25π.
- (b) θ = 0, EnvBP, random start: E = −1.7878 (D = 2), −1.8102 (3), −1.8210 (4), −1.8298 (5), monotonically towards −1.8368(5) (D → ∞, paper) and below the D = 1 value −1.5; m = 0.89, 0.86, 0.84 (D = 2–4). D = 4 took 160 s, D = 5 1.6 h (single thread).
- (c) 80-point scan, D = 4, EnvBP, lowest energy of five starts (FM, FQ, AFQ, SU3 seeds, random), `docs/figures/m3_scan_D4.png`, data `results/m3_scan_D4.jsonl`: AFM3 0–0.225π (m 0.84 → 0.48), AFQ 0.25–0.475π (m = 0, |Q| up to 1.12), FM 0.5–1.25π (m = 1, |Q| = 1/√3), FQ 1.275–1.875π (m = 0), AFM3 from 1.9π (m = 0.84). Jumps in m and |Q| at π/4, π/2, 5π/4 and between 1.875π and 1.9π (paper: θ_c = 1.873(7)π). SPEC asks for D ≤ 6; only D = 4 is scanned so far (D = 5 takes ~1.6 h per point).

NTU follow-up (2026-10-18): the warm-start NTU scan drifted above EnvBP. In the FM phase the cause is the default `initialization="EAT_SVD"` of `truncate_`, which keeps metric-null components after path gates (docs/yastn_findings.md, NTU section); `evolve` now passes `initialization="EAT"` for NTU, and FM states stay exact over 400 steps. In the AFM3/AFQ phases NTU remains 1e-3 to 1e-2 above EnvBP at D = 4 with either initialization (at θ = 0.25π: EnvBP 1.67401, NTU from the BP state 1.68802 (unconverged when stopped), NTU from a product state 1.69106; the gap is open). The 42 partial NTU scan runs used the faulty setting and are not kept.

## M4 — Full update + AD (in progress, 2026-10-18)
AD part: `variational.py` (`Chain`: L-BFGS over raw site-tensor data, tensors normalized inside the loss, energy via `measure.class_energies` on torch tensors; checkpoint/resume; `validate` with the numpy backend at χ = D², 2D², 3D²; `pad`, `convert`), `scripts/m4_variational.py` (one chain, resumable), `tests/test_variational.py`. Design and deviations: docs/ad_workflow.md (**[impl]** marks).

Findings:
- At D = 3 the CTM projector SVDs keep an exactly degenerate pair (~2e-4 σ_max). `fp_ctmrg`'s implicit gradient was wrong by ×10 on one such state; the default is now plain backpropagation through the CTM sweeps (`ad="unrolled"`), with the pair's 0/0 term dropped (`exact_svd_backward(kept_degenerate="zero")`, small pairs only). AD = FD to 2e-8 there, but the dropped term has no a-priori bound, so `Chain` FD-checks the gradient after steps that zero a pair (docs/yastn_findings.md U4b).
- torch L-BFGS with `max_iter=1` allows no line-search zoom by default (`max_eval`); fixed in `Chain` (docs/ad_workflow.md §5b).

First result, θ = π/4, √3×√3, dense: D = 3 AD from the EnvBP state gives E = 1.67403 (validated, χ-converged), 0.0145 below EnvBP (1.68856); C₃ spread 1.8e-3. D = 2 stays at the BP minimum (1.84010).

Symmetry switch (SPEC §4.5): `operators.spin1(sym)` for sym in dense, U1, U1xU1, Z2xZ2. U1xU1 and Z2xZ2 use `operators.SymSpin1`, which builds every operator from its canonical-basis matrix (S± has no definite charge there) and keeps only charge-neutral order-parameter components (U1xU1: Sz, Q2; Z2xZ2: Q1, Q2). U1xU1 charges are (Sz, Sz²), the SU(3) Cartan subgroup, so only J = K bond Hamiltonians are accepted; Z2xZ2 uses the Cartesian basis |x⟩, |y⟩, |z⟩ with π-rotation parities. `states.to_dense` converts a symmetric PEPS for cross-checks; `scripts/m4_variational.py --sym ... --seed bp:NAME` seeds a symmetric chain by EnvBP from a compatible product state (symmetric tensors are not padded). Tests: `tests/test_symmetry.py` (D = 1 energies, D = 2–3 agreement with dense, AD gradient vs FD). At D = 2 dense breaks U1×U1 from the SU3 seed (docs/yastn_findings.md, Symmetric configs).

D = 4 at θ = π/4: the first chains (U1×U1 and dense, χ = 32) reached 1.6592 / 1.6614 and then stopped: a plain χ cut through a multiplet made CTM cycle (U1×U1) or split a near-degenerate pair (dense). Fixed with a multiplet- and gap-aware projector mask (`yastn_compat.gap_safe_truncation`, docs/yastn_findings.md); the chains were restarted from their last good states. Restarted U1×U1 chain (χ = 32, 300 steps, no failed steps, stopped at the step limit): validated E = 1.651264 (χ = 16), 1.650929 (32), **1.650875** (48), CTM converged at each χ, C₃ spread 1.4e-3; 0.0232 below D = 3 (1.67403), still above the paper's 1.632 ± 0.005. The last ~220 steps gained only 5e-5 (|g| ~1e-3), so the chain has plateaued rather than converged. The restarted dense chain stopped at step 80 (E_opt = 1.65588, not validated): from step 61 on the zero-mode gradient repeatedly failed the FD check (AD vs FD up to 5× apart at step 81, four retries), and `Chain` raised. Both restarts predate the single global cut of PR #10 (they ran the two-pass mask); whether the current mask removes the dense failure is untested.

Open (SPEC §6 M4): EnvCTM full update; the π/4 acceptance with U1×U1 (1.632 ± 0.005); D ladder and θ = 0 extrapolation; FQ/AFM3 hysteresis at 1.86–1.89π; gradient gate near θ ≈ 1.87π; profiling at D ≥ 4 (unrolled memory).
