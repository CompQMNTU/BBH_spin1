# YASTN findings (M0)

yastn `1.6.2.dev320+g7ed8ad0e6`, built with
`uv run --no-project --with "yastn @ git+https://github.com/yastn/yastn@7ed8ad0e6" --with numpy,scipy`.
Line numbers refer to that version. Checked 2026-10-14; re-check if the pin in `pyproject.toml` changes.
Regression tests (M0): `tests/test_routing.py` (U1), `tests/test_measure.py` (U2, U3 smoke, D=1 SU(3) energy), `tests/test_yastn_canaries.py` (pinned upstream behavior incl. bugs), `tests/test_fp_ctmrg.py` (U4; needs `--extra ad`), `tests/test_yastn_compat.py` (our workarounds). Run everything with `uv run pytest -m "slow or not slow"` (~40 s).

## API (summary of SPEC §1)

| Item | Result | Where |
|---|---|---|
| `Spin1` dense basis (+1,0,−1); U1 charge = sz | confirmed; U1 `to_numpy` orders charges −1,0,+1 | `operators/_spin1.py:46` |
| `TriangularLattice(dims, boundary, full_patch)` | infinite: confirmed, `site2index=(y−x)%3`, diagonal `Bond(s_b, s_r)`; `full_patch=True` accepts (2,2),(2,3),(4,4),(6,6), CTM runs on (2,2) | `tn/fpeps/_geometry.py:359`, `:388` |
| `TriangularLattice(boundary='obc')` | **broken**: `full_patch=True` → `bonds('d')` has `Bond(None, …)`; `full_patch=False` ignores `dims` (3 sites, bonds to `Site(0,3)`) | `_geometry.py:365-370` |
| `RectangularUnitcell` | no diagonal bonds; `bonds('d')` silently returns h+v | |
| `Gate(G, sites)`, `fill_eye_in_gate` | confirmed | `_gates_auxiliary.py:22`, `:135` |
| `gate_nn_exp(step, I, H, bond=None)`, `gate_local_exp(step, I, H, site=None)` | confirmed | `gates.py:51`, `:67` |
| `yastn.fkron` | confirmed | `tensor/_contractions.py:832` |
| `distribute(..., symmetrize=True)` | appends the reversed list of its own gates only; iterates `geometry.bonds()` so it emits bare diagonal gates on TriangularLattice | `gates.py:194`, `:225` |
| `evolution_step_(..., method='mpo'|'nn')` | confirmed (case-insensitive); `EnvBP` works (has `bond_metric`) though the docstring omits it | `_evolution.py:53`, `_env_bp.py:352` |
| `EnvCTM(psi, init, leg)`, `iterate_(opts_svd, moves, method, max_sweeps, corner_tol)` | confirmed; default method `'2x2 corner'`, `'2x2'` accepted | `_env_ctm.py:44`, `:862` |
| `measure_2x2`, `measure_nsite_exact` (any rectangle), `measure_2site`, `transfer_matrix_spectrum` (numpy, scipy `eigs`) | confirmed; 2×3 knight window runs at small D | `_env_ctm_measure.py:293`, `:539`, `:586` |
| `measure_nsite_exact_oe` | **does not exist** (full method list) | `_env_ctm.py:1021` |
| `measure_rdm_nn`, `measure_rdm_diag`, `measure_rdm_2x2` | **crash on dense**: `IndexError: tuple index out of range` from `O0.n[0]`; low-level `rdm1x2`, `rdm2x1`, `rdm2x2_anti_diagonal` work | `envs/rdm.py:750`, `:796`, `:869` |
| `fp_ctmrg`, `NoFixedPointError` | exist; `fp_ctmrg` returns only `env` (docstring says tuple); torch imported at module level | `envs/fixed_pt.py:612`, `:36`, `:22` |
| `to_dict` / `from_dict` | round-trip a TriangularLattice PEPS; `save_to_dict` deprecated, `load_from_dict` works | |
| Custom Z2xZ2 | `sym_abelian` subclass via `make_config(sym=cls)` works (SVD tested); `'Z2xU1'` not accepted as string | `tensor/_initialize.py:123` |

**Defaults that iterate over all bonds raise on TriangularLattice** with
`YastnError: Site(1, 0), Site(0, 1) are not nearest-neighbor sites.`:
`measure_nn()`, `EnvBP.measure_nn()`, `truncate_(bond=None)` (`_evolution.py:229`), `Peps.get_bond_dimensions()` (`_peps.py:139`), gates from `distribute`.

## U1 — gate routing (resolved)

- A bare 2-site diagonal gate raises the error above (EnvBP and EnvNTU).
- A 2-site gate with `sites` = an NN path was compared with exact ED on a 2×3 OBC entangled state. Overlap with the exact state was 1.00000000000000 for:
  - both 3-site diagonal paths,
  - the (1,1) NNN path,
  - all three 4-site knight paths,
  - with EnvBP and EnvNTU, in `mpo` and `nn` modes.
  Overlap with wrong placements (first or middle site) was 0.86–0.91, so the test is discriminating.
- Every bond along the path is truncated (truncation error ~1e-15 in this test).
- On the |+1,0,−1⟩ product state, the maximum deviation from the exact 2-site result was 6e-17. The middle site stayed in |0⟩ (weight off |0⟩ ~1e-17).

## U2 — diagonal measurement (resolved)

- `measure_nn` raises; `measure_rdm_diag` crashes (dense); `measure_line` raises `Sites should form a horizontal or vertical line.`
- `measure_2x2`, `measure_nsite_exact` and a manual `rdm2x2_anti_diagonal` contraction agree: 0.5547073039441632 / …628 / …629.

## U3 — knight-move windows (partly)

`measure_nsite_exact` runs on a 2×3 window at small D. Cost at the target D, χ is not yet measured.

## U4 — `fp_ctmrg` on TriangularLattice (resolved with a patch)

Setup (`tests/test_fp_ctmrg.py`): torch backend, float64, √3×√3 TriangularLattice, dense, D=2 random tensors (normalized), loss = Σ_diag ⟨Sz Sz⟩ via `rdm2x2_anti_diagonal`, `fp_ctmrg` with `policy='fullrank'`, `corner_tol=1e-10`. Check: directional derivative along a random direction, AD vs central finite differences (FD stable to 1e-8 for ε = 1e-3…1e-5). One forward+backward ≈ 4–9 s.

- **Runs:** forward converges, backward series converges, gradient finite. Returns only `env`.
- **Bug (upstream):** `backend/linalg/torch_svd_gesdd.py:4`, `safe_inverse(x, eps_abs)` overwrites its argument with `eps_abs=1.0e-12` and returns x/(x²+1e-12). The scale-relative `scaled_eps = Smax²·ad_decomp_reg` computed in `SVDGESDD.backward` is never used. So 1/σ and 1/(σᵢ±σⱼ) are damped whenever σ or a gap is below ~1e-6 in **absolute** size. The CTM projector SVDs in the test have σ_max ≈ 7e-2, σ_min down to 1e-22 and exact-zero gaps.
- **Not the cause:** truncation. A truncated `svd_with_truncation` with O(1), well-separated singular values matches FD to 1e-9 (the full SVD is computed first, so the discarded-subspace coupling is present). Not the fixed-point series (plain backprop through `ctmrg_` gives the same AD value), not the triangular geometry (`RectangularUnitcell` gives identical numbers).
- **Evidence** (same state, directional derivative; FD = 0.141684 / 0.141671 / 0.141666 at χ = 4 / 8 / 16):

  | χ | yastn as is | `safe_inverse` with the intended scaled ε | exact inverse (0 at 0) |
  |---|---|---|---|
  | 4 | 1.2e-2 | 1.9e-6 | 4.4e-8 |
  | 8 | 2.0e-2 | 1.3e-6 | 7.2e-8 |
  | 16 | 4.0e-1 | 9.5e-4 | 7.5e-8 |

  Isolated SVD (12×9, spectrum scale·0.5ᵏ, keep 6): exact at scale 1, wrong sign at scale 1e-5.
- **Workaround:** `bbh_tri.yastn_compat.exact_svd_backward()` (context manager) / `patch_svd_backward()` replace `SVDGESDD.backward` by `svd_backward`: same formula, exact 1/σ, 1/(σᵢ+σⱼ), 1/(σᵢ−σⱼ), each 0 only at exactly 0. `backward()` must run inside the context.
- **Degenerate singular values (Codex review of PR #4).** For a kept pair with |σᵢ−σⱼ| at roundoff level, the 1/(σᵢ−σⱼ) term of a gauge-invariant loss is 0/0 with a finite, nonzero limit. Roundoff destroys it, and no formula that sees only the incoming gradients recovers it. Measured on a 8×6 matrix, kept pair at 0.7, 6 seeds:

  | gap | exact inverse (this patch) | gaps set to 0 below 1e-12·σ_max |
  |---|---|---|
  | 0, 2⁻⁵² | 1e-2 … 8e-1 | 1.6e-1 |
  | 1e-14 | 1e-4 … 1e-2 | 8e-2 |
  | 1e-12 | 1e-6 … 6e-4 | 8e-2 |
  | ≥ 1e-9 | ≤ 1e-7 | ≤ 1e-7 |

  So `svd_backward` keeps the exact inverse (0 where x = 0 or 1/x would overflow, so discarded subnormal tails cannot produce inf·0 = NaN) and raises `DegenerateSVDWarning` **whenever any gradient reaches a singular vector** of (a) a pair with gap ≤ 1e-10·σ_max **of that block**, or (b) a σ, gap or σᵢ+σⱼ whose reciprocal is undefined or overflows; it also warns if the result is not finite. There is no tolerance. Discarded columns get exactly zero gradient, so CTM tails (σ ≈ 1e-20, zero gaps) never trigger it: no warning at χ = 4–16.
  History, from six Codex review passes of PR #4. Each fraction- or magnitude-based version had a counterexample, so the test is now structural:
  - skew parts as the measure: they cancel exactly for a symmetric incoming gradient (A = diag(1, .7, .7, .3), keep 3, L = R₁₂ + R₂₁: AD 0 vs FD 2);
  - all-block σ_max: yastn passes the maximum over symmetry blocks, which flagged a well-resolved block (2e-12, 1e-12) next to a norm-1 block. The scale is now block-local;
  - raw Uᴴ·gU weights: they carry a factor σ, hiding small pairs (diag(1, .7, 1e-8, 1e-8): dL/dA₂₃ = 0 vs 1); a 1/σ rescaling with an ε·σ_max floor then hid a pair at 1e-22;
  - any fraction of a total: a projector term on small singular values inflated the denominator (diag(1, .7, .7, 2e-8, 1e-8, 1e-9), L = R₁₂ + R₂₁ + P₃₄ + P₄₃: AD 0 vs 1);
  - overflow: diag(1, 1e-300, 1e-300 − 1e-310), keep 1: discarded gap ~1e-310, 1/gap = inf, NaN gradient. Then diag(1, 1.2e-308, 6e-309): finite F and G overflow in F + G. Fix: 1/σ, 1/gap and 1/sum are formed only for singular vectors (pairs) that receive gradient; all others are exactly 0;
  - an active, well-resolved pair at the float64 limit (diag(1e-307, 9.9e-308)): 1/gap overflows and was dropped silently. Now an active pair with an unusable 1/gap or 1/sum also warns, as does any non-finite result.
  **Undetectable case: exactly zero kept singular values.** gU and gV carry a factor σ, so for σ = 0 nothing about the lost term reaches the backward, and kept zeros look like discarded ones (`test_degenerate_zero_pair_is_undetectable`, strict xfail). Hence AD runs must truncate with `opts_svd['tol'] > 0` so that no zero or roundoff-level singular value is kept (our CTM runs use tol = 1e-8). Known false positive: a loss that depends on a degenerate pair only through its span (a projector onto both vectors).
- **Tests:** `test_fp_ctmrg_gradient_matches_finite_difference[8,16]` (patched, rel 1e-5, `DegenerateSVDWarning` is an error), `test_fp_ctmrg_gradient_wrong_without_patch` and the SVD canary in `test_yastn_canaries.py` (pin the upstream bug), `tests/test_yastn_compat.py` (small σ, resolved and degenerate pairs, each Codex counterexample, the exact-zero limitation).
- Reported upstream: not yet.

## U4b — AD through CTM with a kept degenerate pair (M4, 2026-10-18)

Setup: θ = π/4, √3×√3, dense, D = 3 EnvBP states, χ = 18, loss = E/site (`bbh_tri.variational`), directional derivative AD vs central FD (FD stable to 1e-8 over ε = 1e-3…1e-5).

- **The CTM projector SVDs keep an exactly degenerate pair**: σ₁ = σ₂ ≈ 1.3–2.3e-4 σ_max, both vectors kept, in every CTM move. It survives 1e-2 random noise on the site tensors, so it is structural, not the SU(3) point. With the default `svd_backward` this raises `DegenerateSVDWarning` on every gradient at D = 3 (D = 2: no degeneracy).
- **Kept pair: zeroing is an empirical approximation, checked by FD at run time.** The pair's 1/(σᵢ − σⱼ) term is a 0/0 limit; it vanishes only if the loss does not depend on how the pair splits, which CTM projectors (S^(−1/2)) do. `exact_svd_backward(kept_degenerate="zero")` drops it for pairs whose two vectors both receive gradient and whose σ ≤ `ZERO_MAX_SIGMA` = 1e-3 σ_max of the block; larger kept pairs and pairs straddling the cut still warn. Measured error of dropping it: generic gauge-invariant SVD loss 1e-2 (pair at 0.7 σ_max), ~1e-3 (1e-2), ~5e-6 (1e-4) relative (`tests/test_yastn_compat.py`); CTM energy at D = 3 with the pair at 2e-4 σ_max, plain backpropagation through the sweeps (`Chain(ad="unrolled")`): AD = FD to 2e-8 (`tests/test_variational.py`). The error has no bound in terms of the pair's size: it is the loss's sensitivity to mixing within the pair, which can be O(1) (Codex, PR #8: L = R₁₂ + R₂₁ of the reconstruction of diag(1, 1e-4, 1e-4) has derivative 2, the zero mode gives 0; `test_zero_mode_error_is_not_bounded_by_pair_size`). So every zeroed pair is counted (`yastn_compat.zeroed_pairs()`) and `Chain` compares AD with central differences after a step that zeroed one (first time, then every `fd_check_every` = 10 steps), along the gradient and two random directions (`variational.fd_check`; tolerance 1e-3 |g| + energy noise / FD step), failing the step otherwise. The random directions matter: a dropped component orthogonal to the surviving gradient is invisible along the gradient (Codex, PR #8: εR₀₀ + R₁₂ + R₂₁; `test_fd_check_catches_component_orthogonal_to_the_gradient`). A degenerate pair among the leading CTM singular values (e.g. a symmetric point with multiplets at the top of the spectrum) is not covered; use symmetric tensors there, where the pair falls into different blocks.
- **`svd_tol` must sit well above roundoff.** With CTM `tol = 1e-10` a pair at 1.8e-10 / 9.4e-11 σ_max straddled the cut (gap below 1e-10 σ_max) and warned. `tol = 1e-8` removed it.
- **`fp_ctmrg` (implicit fixed-point backward) can be wrong on such a state.** On a fully converged D = 3 EnvBP state it missed FD by a factor 10 (AD 0.145 vs FD 0.0144; AD also changed with `corner_tol`, FD did not), with or without zeroing. On a less converged D = 3 state with the same degeneracy it was right (3e-6). Presumably the gauge fixing of the fixed point is not unique on a degenerate spectrum. Not pinned by a canary (no cheap reproducer); the driver defaults to `ad="unrolled"`.
- **`fp_ctmrg` forward at `corner_tol = 1e-10`** did not converge in 500 sweeps on the D = 3 state (`NoFixedPointError`); 1e-9 does.

## U4c — `fp_ctmrg` applies `truncation_f` everywhere (M4, 2026-10-26)

At the pinned commit `fp_ctmrg` forwards every key of `ctm_opts_fwd` besides `method`, `max_sweeps`, `opts_svd` and `corner_tol` to `EnvCTM.ctmrg_`, and copies `ctm_opts_fwd` (updated by `ctm_opts_fp`) into the options of the gauge-fixing CTM step and of the backward fixed-point iteration (`FixedPoint.fixed_point_iter`, `env.update_`). `proj_corners` turns `truncation_f` into `opts_svd['mask_f']`, so one mask sets the cut in the forward, the gauge fixing and the backward. Only the multi-GPU forward (`fwd_devices` with more than one device, `iterate_D_`) passes `truncation_f=None`. The backward runs the CTM step under `torch.func.vjp`, whose wrapped tensors have no storage: `.numpy()` raises `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`, while `.tolist()` and `.item()` work, so a mask must read the spectrum with `tolist` (`gap_safe_truncation` does). Canary: `test_yastn_canaries.py::test_fp_ctmrg_passes_truncation_f_to_forward_and_backward` (a keep-5 mask at `D_total` = 8 gives 5×5 corners and is called again in the backward). `Chain(ad="fp")` passes its `gap_safe_truncation` mask this way (issue #22).

## CTM truncation at D = 4: cut through a multiplet (M4, 2026-10-18)

Two AD chains at θ = π/4, D = 4, χ = 32 stopped after 4 failed steps each, both at line-search trial points (reproduced from the checkpoints):
- **U1×U1:** CTM never converged; the corner spectra alternated with max_dsv ≈ 0.028 for 500 sweeps (`NoFixedPointError`). The same state in dense cycles too (2.5e-2 / 1e-4 on alternate sweeps), so it is not a symmetry artifact. χ = 48 or 64, or `tol` = 1e-6, converge in 11 sweeps (E changes by 7e-6). The χ = 32 cut falls inside a multiplet that spans several symmetry sectors, and the CTM alternates between keeping one part or the other.
- **Dense:** the gradient warned: singular values near the `tol` = 1e-8 cut (≈1e-8 σ_max) are closer than 1e-10 σ_max, so the cut kept one member of a near-degenerate pair and discarded the other.
- **Fix:** `yastn_compat.gap_safe_truncation(tol, chi, eps_multiplet)`, passed as `EnvCTM.iterate_(truncation_f=...)`. The kept set is a prefix of the globally sorted spectrum; the cut moves down until, at the same position, (a) the relative gap between the last kept and first discarded value exceeds eps_multiplet (whole multiplets across sectors) and (b) in every block the last kept and first discarded value differ by more than 10 × the backward's degeneracy threshold × block max. (A first version applied (b) as a per-block backoff after (a), which could split a cross-sector multiplet again: Codex, PR #10; now a test.) On the trial points: CTM converges in 11 sweeps for eps_multiplet = 1e-6…1e-3 (29–31 states kept), and the dense gradient no longer warns (E unchanged to 1e-14). `Chain` (default `eps_multiplet` = 1e-6) and `validate` use it; `validate` now reports `ctm_converged` per χ. Regression: `tests/test_yastn_compat.py::test_multiplet_safe_cut_stops_ctm_cycle` (fixture `tests/data/d4_pi4_ctm_cycle.npz`).

## NTU: default initialization keeps metric-null junk on path gates (M3 follow-up, 2026-10-18)

- **Symptom:** NTU warm-started from exact FM states (θ = 0.7π, D = 4, τ = 0.01) drifted up by 2.6e-4 over 400 steps; per-step truncation error grew from 5e-6 to 2e-4 (EnvBP: 5e-11).
- **Cause:** after a 3-site path gate (our triangle MPO, or a 2-site gate routed through a middle site), the first path bond is truncated while the middle site still carries the gate's virtual leg. Components that the next site annihilates are null in the NTU metric (60–80% of its eigenvalues are below the error threshold), so any amount of them costs zero truncation error. `truncate_(initialization="EAT_SVD")` (default) tries both and keeps the smaller error; the SVD branch (unweighted SVD of R₀R₁, error 6e-17) beats EAT (2e-10) by roundoff and keeps rank 3 with singular values ~0.1 of the leading one, which `opts_svd['tol']` cannot remove. On the exact D = 1 FM eigenstate one gate grows the bond 1 → 3 (`test_ntu_default_initialization_inflates_bonds_on_path_gates`). Plain 2-site gates are unaffected, as is EnvBP.
- **Fix:** `initialization="EAT"`: FM held exactly for 400 steps (truncation error ~1e-10, bonds stay D = 1). No effect elsewhere: at θ = 0.1π and 0.25π (D = 4, warm start from EnvBP) EAT and EAT_SVD give the same energy to 8 digits. Not fixed: `fix_metric=1`, `opts_svd['tol'] = 1e-4`, `initialization="SVD"`, `method="NN"` (splitting the MPO into 2-site gates still leaves the virtual leg open).
- **Separate, open:** in the AFM3 and AFQ phases NTU stays 1e-3 to 1e-2 above EnvBP at D = 4 (both with truncation errors 2–5e-3 per step); see docs/milestones.md (M3).

## Serialization

- `TriangularLattice.to_dict()` (`_geometry.py:409`) stores only `pattern`; `Peps.from_dict` calls `TriangularLattice(**d['geometry'])`, which ignores it. A `full_patch=True` (9-site) state reloads as the 3-site √3×√3 cell, an OBC 2×3 `full_patch=True` state (6 sites) as infinite 3×3, silently. (OBC needs `full_patch=True` anyway; without it `dims` is ignored, see the API table.) Workaround: `bbh_tri.yastn_compat.peps_to_dict(psi)` stores `dims`, `boundary`, `full_patch` instead; `Peps.from_dict` then rebuilds the right geometry (tested for default, full_patch and OBC). The default √3×√3 cell round-trips either way.
- `TriangularLattice(full_patch=True).bonds()` (no direction) raises `TypeError` (tuple + list, `_geometry.py:407`). Iterate `bonds('h')`, `bonds('v')`, `bonds('d')` separately.

## Symmetric configs

A U1-symmetric `Spin1` on TriangularLattice with path gates, NTU at D=3, and then CTM runs without errors.

M4 (2026-10-18), `operators.SymSpin1`:
- A custom `sym_abelian` subclass (`sym_Z2xZ2`, fusion mod 2 on both charges) passed as `make_config(sym=cls)` works through gates (`eigh`, `svd_with_truncation`), EnvBP evolution, CTM, `Peps.to_dict`/`from_dict` and torch AD through CTM (gradient = FD to 1e-5 rel, `tests/test_symmetry.py`). `U1xU1` is built in.
- PEPS site tensors have legs (t, l, b, r, p) with p a fused pair (physical, ancilla) (`hist=p(oo)`); the ancilla is one-dimensional and `product_peps` puts the charge of a charged site vector on it, so the tensor charge stays `n = 0`. A tensor lists only the sectors of its own nonzero blocks, so `to_numpy()` without `legs` drops absent physical states and can give the two ends of a bond different sizes; `states.to_dense` expands over the full physical space and the union of both bond ends.
- At small D a dense truncation can break a symmetry the dense run started with: from the SU3 product state at θ = π/4, dense EnvBP at D = 2 develops U1×U1-violating rdm elements ~1e-3 and ends at 1.8401, the U1×U1 run at 1.8774. At D = 3 both give 1.6886583 (1e-11). U1 (from FQ at 1.6π and from SU3 at π/4) and Z2×Z2 (from AFQ at 0.35π) agree with dense at D = 2.
- Small-D cost: U1×U1 EnvBP at D = 3 took 64 s vs 22 s dense for the same run (1×1 blocks, many sectors); the payoff is expected only at large D.

## Sanity run at the SU(3) point, θ = π/4 (NN only)

h_bond = cos θ S·S + sin θ (S·S)² = (P_swap + 1)/√2, checked as an operator identity (also Σh = (ΣP + N_b)/√2 on the cluster).
Gates: 2-site path gates at step τ/2 with the reversed list appended by hand (second order); τ = 0.1 → 0.05 → 0.02.
Initial state: |+1⟩,|0⟩,|−1⟩ on the three sublattices plus small noise. Infinite √3×√3 cell, dense.

| Case | Setting | E/site | per bond h / v / d | max trunc. err. |
|---|---|---|---|---|
| ED, 3×3 OBC (16 bonds) | exact, E0 = 6.0015950 | 0.6668438888 | 0.40114 / 0.40114 / 0.29699 | — |
| D=1 product | χ=1 | 2.121320343559643 (= 3/√2, err 4e-16) | 0.70710678 each | — |
| D=2, EnvBP | χ=4, 6 | 1.87946350 | 0.5902 / 0.5904 / 0.6989 | 6.7e-3 |
| D=2, EnvNTU (**partial**, stopped after τ=0.05, 300 steps) | χ=4 | 1.84256 | 0.506 / 0.531 / 0.806 | 1.5e-2 |
| D=3, EnvBP | χ=9 / 14 | 1.68856206 / 1.68856202 | 0.5620 / 0.5621 / 0.5644 | 2.5e-3 |
| D=3, EnvNTU (**stopped**, 5 steps only) | χ=9 | 1.7075 | — | — |

Reference: Niesen–Corboz D→∞ 1.630(1); best finite-D 1.632 (D=12, variational update, abelian symmetries; no SU(3)-symmetric tensors, see SPEC §3).

Notes:
- The ED row is an operator check only. With 16 bonds on 9 sites it is not comparable per site with the infinite lattice.
- D=3 restores C₃ (h, v, d within 0.4%). D=2 does not: the diagonal bond is much higher and ⟨Sz⟩ = ±0.76–0.81, so the state is likely stuck in an anisotropic configuration. Not investigated; treat D=2 as unreliable.
- CTM did not reach `corner_tol=1e-11` in 300 sweeps on the final runs (remaining change 1e-7 to 1e-9). At D=2, χ=6 reported max_D=4.

## Performance

- **NTU with untruncated path gates:** a path gate through a bond that already has D≈9 grows it toward 81, and NTU's metric optimization then takes minutes (knight path (1,0)→(1,1)→(0,1)→(0,2) after D=9 entangling: >5 min). Keep D small in such tests; in production every path bond is truncated anyway.
- **yastn bug:** `evolution_step_(..., max_iter=0)` raises `UnboundLocalError: cannot access local variable 'error2'` in `optimize_truncation` (`_evolution.py:852`).
- **Warning (harmless):** with the torch backend, `Spin1` construction casts complex values to real (`backend_torch.py:230`).

- Default threading on 48 cores: D=2 NTU took ~10 s per step. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Single-threaded, D=3 NTU did not finish 100 steps in 8 min. Profile NTU before M3; EnvBP is the practical small-D default.
