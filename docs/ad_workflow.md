# Ground-state search by implicit-AD CTM (design)

Status (2026-10-18): implemented in `src/bbh_tri/variational.py` (`Chain`, `validate`, `pad`, `convert`) and `scripts/m4_variational.py`; tests in `tests/test_variational.py`. Deviations from the design below are marked **[impl]**. Not yet implemented: warm-started environments, Adam fallback, HDF5 output. (Since implemented: multiplet-aware truncation, symmetric tensors, the config layer: `bbh-run`, docs/milestones.md "Production layer".)
API facts are from yastn `1.6.2.dev320+g7ed8ad0e6`: `envs/fixed_pt.py`, `envs/rdm.py`, and the upstream test `tests/peps/test_fp_tV_complex.py`, which is the reference usage pattern.
See `docs/yastn_findings.md` for the related M0 results.

## 0. What `fp_ctmrg` gives us

- **Forward:** runs CTMRG to convergence on the passed `EnvCTM`. Raises `NoFixedPointError` if it doesn't converge within `max_sweeps`/`corner_tol`, or if symmetry sectors change. Then does one extra gauge-fixed CTM step (`'fullrank'` SVD only).
- **Backward:** sums the series Σₙ (∂f/∂C)ⁿ using the vector-Jacobian product of that single step. It stops when the terms fall below `corner_tol`, when the change stops decreasing ("stall"), or at `max_sweeps`.
  - The gradient is therefore approximate. Its quality has to be checked, not assumed (§5).
  - It prints `full grad diff` to stdout every iteration.
- **Return value:** only `env`. Unset fixed-point options are inherited from the forward options.

## 1. Pipeline overview

```
config ──► seeds ──► [optimize chain]×N ──► validate (numpy, larger χ) ──► select ──► D ladder / θ scan ──► extrapolate
                         ▲   │ checkpoint every step
                         └───┘ retry on NoFixedPointError
```

One **chain** is one run with a fixed (model params, unit cell, D, χ_opt, symmetry, seed). Chains are independent and run as SLURM array tasks. Selection and extrapolation are post-processing.

## 2. Config (one chain)

```yaml
model:    {theta: 0.25pi, J2: 0.0, K2: 0.0}
cell:     {type: sqrt3, dims: [3, 3], full_patch: false}
psi:      {D: 3, sym: dense, dtype: float64}   # complex128 only if a seed needs it
seed:     {kind: from_file|product|bp|pad|random, path: ..., noise: 1.0e-3, rng: 17}
ctm_opt:  {chi: 3D², method: 2x2, corner_tol: 1.0e-8, max_sweeps: 200,
           svd_policy: fullrank, multiplets: true, init: warm|dl|eye}
optim:    {method: lbfgs, max_steps: 300, history: 20, line_search: strong_wolfe,
           tol_E: 1.0e-8, tol_E_window: 10, tol_grad: 1.0e-6}
validate: {chis: [2D², 3D², 4D²], corner_tol: 1.0e-10, max_sweeps: 500}
runtime:  {threads: 1, device: cpu, checkpoint_every: 1}
```

χ_opt = 2D² is a starting point. The validation stage (§4) decides whether that was enough.

**[impl] χ_opt = 3D² (2026-10-20).** At 2D² the optimization stalls near convergence. Measured at a U1×U1 D = 5 state:
- The AD slope along −g matches central differences exactly for h ≤ 3e-5.
- For h = 1e-4, 3e-4 and 1e-3 it misses them by 23%, 8% and 2%, always by the same energy offset of ~2e-7: a step discontinuity of the loss along the line.
- Running the CTM to 200 sweeps instead of 14 leaves the energies unchanged, so this is not convergence noise.
- At χ = 75 (3D²) AD and FD agree to 4 digits for every h, and the optimization energy moves by 1.1e-5 (1.6493344 → 1.6493235).

The cause: at 2D² the cut lies inside a near-degenerate group (relative splitting < `eps_multiplet`). `gap_safe_truncation` keeps or drops the group whole, so the kept set changes discretely where the grouping or the χ boundary changes along a search direction. Near convergence the jumps (~1e-7) match the per-step gains, so strong-Wolfe line searches take many evaluations (D = 6: 19 for one step) and progress crawls (D = 4: ~1.6e-7 per step at |g| ~ 1e-3).

Tested and ruled out at D = 4: resetting L-BFGS, renormalizing the tensors to unit norm, history 100. Each did worse than continuing. In U1×U1 at D = 5, a step at χ = 3D² costs about the same as at 2D² (~175 s): the environment rank is set by the spectrum, not χ.

`scripts/m4_variational.py` now defaults to 3D².

**[impl] Results at 3D² (2026-10-21), U1×U1, θ = π/4.** 3D² shrinks the jumps but does not remove them.
- D = 5 (χ = 75) converged: 274 steps, final |g| = 7e-5. Validated E = 1.6490700 (χ = 75), 1.6490691 (χ = 100).
- D = 4 (χ = 48) stalled after 15 steps at |g| = 4.18e-3: 8–14 evaluations per step, E flat to 1e-10. At the stalled state:
  - Along −g, the AD slope matches the energy change for steps t ≤ 1e-5. The best point is t ≈ 1e-4 with dE = −5.5e-7, and E rises from t ≈ 3e-4 on. The 1e-4 point lies below the quadratic through the others: the loss jumps by ~3e-7 along the line.
  - A tighter `corner_tol` (1e-11) changes the gradient by 5e-10 (relative). Random displacements of 1e-5 change it by 20% and E by 3e-6; at 1e-3, cos(g, g′) = 0.25.
  - So the gradient is right but describes the loss only within ~1e-5; the per-step gain left (≲1e-6) is below the jumps. The χ-dependence of E is larger: 1.6508657 (χ = 48) vs 1.6508443 (χ = 64).
  - Restarted at χ = 64 (4D²) from that state, the chain moves again with one evaluation per step: E (at χ = 64) 1.6508440 → 1.6508416 in 8 steps, gains shrinking from ~3e-7 to ~1e-7 per step. The jumps shrink with χ; the remaining gain (~1e-6) is small next to the χ-dependence, so for the D extrapolation report the largest-χ validation.
- The dense D = 4 chain (GPU) stalled the same way at |g| = 4.2e-3, but 9e-5 higher (validated 1.6509549 at χ = 48).

**[impl] D = 6 at 3D² (2026-10-24), U1×U1, θ = π/4.** The chain seeded from the best D = 5 state stalled at step 172 (|g| = 2.5e-3, gains ~1e-9 per step). Line scan along −g at χ = 108: dE = +9.6e-6, +9.3e-6, +9.2e-6, +1.8e-5, +1.6e-4 at t = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, against −|g| t = −2.5e-7 … −2.5e-5. Every step lands ~9e-6 higher: the optimizer's state sits on a low branch of the cut, and the fresh validation at χ = 108 (1.6408868) lies 8.9e-6 above its E₀ (1.6408779). At D = 6 the jumps at 3D² are ~100× those at 2D², D = 5, so 3D² does not prevent stalls there; the remaining gain is far below the D-extrapolation error, and the validated energy is the number to use.

## 3. Inner loop (closure evaluated by the optimizer)

Parameters are the raw data `t._data` of the unique site tensors (3 for √3×√3), with `requires_grad=True`. This is the same pattern as the upstream test.

1. **Build the state.** Copy the template tensors, insert the parameters, and normalize each tensor by its norm (inside the graph). Build `Peps(geometry, tensors)`.
2. **Environment.**
   - `warm` (default after step 0): start from the previous step's converged env with `.detach()` and the new `psi` attached.
   - `dl` / `eye`: cold start, used on retry.
   - Warm start is an untested speed optimization. If it causes trouble, fall back to `dl`, as the upstream test does.
3. **Fixed point.** **[impl]** Default is `ad="unrolled"`: plain backpropagation through every CTM sweep from `init="eye"` (`env.iterate_`), not `fp_ctmrg`. At D = 3 the CTM SVDs keep an exactly degenerate pair, and the implicit fixed-point gradient was wrong by a factor 10 on one such state (`docs/yastn_findings.md` U4b). `ad="fp"` (below) remains available. Unrolled memory grows with the number of sweeps (fine at D ≤ 3; profile at D ≥ 4). CTM options: `corner_tol` 1e-9 (1e-10 did not converge at D = 3), `max_sweeps` 500, `opts_svd tol` 1e-8.
   **`ad="fp"` in production (2026-10-25).** `bbh-run` option `ad: fp` / `m4_variational.py --ad fp`. Memory no longer grows with the sweeps. `max_sweeps` caps the forward CTM sweeps (a forward that does not converge raises `NoFixedPointError`, which retries the step as in the unrolled case). It also caps the terms of the backward fixed-point series, which stops there without an error. The forward truncates with yastn's own `eps_multiplet`, not `gap_safe_truncation`. A checkpoint resumes only with its own `ad`; to switch, seed a new `out` with the old `OUT.pt` (a plain path, not `bp:PATH`, which would re-evolve the state). Gradient check at θ = 0, dense, inside `exact_svd_backward()` (pinned yastn, A100). The columns give AD / FD along −g and two seeded random directions:

   | state | χ | ad | E | along g | random 1 | random 2 | GPU memory |
   |---|---|---|---|---|---|---|---|
   | D = 4 (χ = 48 for 1000 steps, then χ = 64 for 122) | 48 | unrolled | −1.831610486 | 9.7360e-4 / 9.7364e-4 | −1.1905e-5 / −1.1903e-5 | 1.4543e-5 / 1.4544e-5 | 41.6 GB |
   | same | 48 | fp | −1.831610486 | 9.7395e-4 / 9.7363e-4 | −1.1855e-5 / −1.1903e-5 | 1.4452e-5 / 1.4544e-5 | 1.7 GB |
   | same | 64 | fp | −1.831611596 | 8.7140e-4 / 8.7144e-4 | −5.6051e-6 / −5.6028e-6 | 1.9116e-5 / 1.9118e-5 | 2.9 GB |
   | D = 5 (AD step 191) | 75 | fp | −1.833935693 | 3.66335e-3 / 3.66310e-3 | 9.11194e-5 / 9.11195e-5 | −1.74138e-5 / −1.74152e-5 | 8.4 GB |

   All pass `fd_check` (tolerance 1.5e-6 at D = 4, 4.2e-6 at D = 5). At D = 4, χ = 48, the fp gradient differs from the unrolled one by 0.3 % (cos 0.999995). Unrolled at χ = 64 does not fit an 80 GB A100, and D = 5 needs ~300 GB unrolled on CPU. One fp evaluation at D = 5, χ = 75 took 279 s forward and 41 s backward on the GPU. The D = 3 failure above (an exactly degenerate pair on a fully converged state) is not excluded in general: keep `Chain`'s FD check after steps that zero a pair.
   ```python
   env = fp_ctmrg(env,
       ctm_opts_fwd={'method': '2x2', 'corner_tol': tol, 'max_sweeps': n, 'use_qr': False,
                     'opts_svd': {'policy': 'fullrank', 'D_total': chi, 'tol': 1e-10},
                     'truncation_f': multiplet_mask},     # see §6
       ctm_opts_fp={'opts_svd': {'policy': 'fullrank'}})
   ```
4. **Energy per site.** For each unique anchor site (x,y), one `rdm2x2((x,y))` (81×81 for dense spin-1) contains every NN bond exactly once:
   - h (x,y)–(x,y+1)
   - v (x,y)–(x+1,y)
   - d (x+1,y)–(x,y+1)

   It also contains the NNN (1,1) bond (x,y)–(x+1,y+1).

   E/site = (1/N_unique) Σ_anchors [h_h + h_v + h_d + h₂(1,1)], each term normalized by the rdm trace.
   - Knight-move NNN terms (J₂, K₂ ≠ 0 only) need 2×3 / 3×2 windows from `measure_nsite_exact`. Check they are differentiable in torch (U3) before M5; skip them when J₂ = K₂ = 0.
   - Bond classes are accumulated separately for logging.
5. Return E/site (real part).

Per-step log: E, per-class energies, ‖∇‖, CTM sweeps, backward series steps and whether it stalled, wall time for forward and backward.

## 4. Validation (separate pass, numpy backend)

The optimizer's energy is measured at χ_opt with `corner_tol`=1e-8. Never report it directly.

1. Load the tensors and convert them to a numpy config.
2. Run a fresh `EnvCTM` + `ctmrg_` at each χ in `validate.chis` with a tight `corner_tol`.
3. Record:
   - E per class and per site;
   - m and |Q| per sublattice;
   - the C₃ spread max(h,v,d) − min(h,v,d);
   - the correlation length ξ (`transfer_matrix_spectrum`, numpy only).
4. Accept the chain only if all of these hold:
   - |E(χ_max) − E(χ_max-1)| < 1e-5;
   - |E(χ_val) − E(χ_opt)| < 1e-5. A large **rise** at χ_val means the optimizer exploited a too-small environment (χ bias: E(χ_opt) is artificially low); a large drop means χ_opt under-resolves the state. Either way, rerun with larger χ_opt.
   - CTM converged at every χ_val.

## 5. Gradient quality gate (before any production chain)

**[impl] Status (2026-10-18), production loss (E/site, θ = π/4, χ = 2D²):** D = 2: AD = FD to 1e-5 or better with both `ad="unrolled"` and `"fp"`. D = 3: every gradient hits a kept degenerate pair (σ₁ = σ₂ ≈ 2e-4 σ_max); with `exact_svd_backward(kept_degenerate="zero")` (drops the pair's 0/0 term, only for pairs below 1e-3 σ_max) and `ad="unrolled"`, AD = FD to 2e-8 (no a-priori bound on the dropped term, so `Chain` FD-checks steps that zero a pair). `svd_tol` 1e-10 made a near-roundoff pair straddle the cut (warning); 1e-8 does not. Not yet run: θ ≈ 1.87π.

**Status (2026-10-15): passes with the SVD patch.** With YASTN's own SVD backward, AD missed FD by 2% at χ=8 and 40% at χ=16 (absolute 1e-12 regularizer; `docs/yastn_findings.md` U4). Inside `bbh_tri.yastn_compat.exact_svd_backward()` the error is ~1e-7 at χ = 4–16 (D=2, Sz Sz loss; `tests/test_fp_ctmrg.py`). Every forward and backward pass of the optimizer must run inside that context, with `DegenerateSVDWarning` turned into an error (`warnings.simplefilter('error', DegenerateSVDWarning)`); on that error, treat the step like a `NoFixedPointError` (§6 failure handling) or switch to symmetric tensors. CTM `opts_svd` must have `tol > 0` (kept exact zeros give wrong, undetectable gradients). The checks below still apply to the production loss.

- **Finite-difference check** at θ=π/4, D=2, χ=8, √3×√3, dense float64:
  - Use a random direction v and compare (E(x+εv) − E(x−εv))/2ε with ∇E·v for ε ∈ {1e-3, 1e-4, 1e-5}.
  - Acceptance: relative error < 1e-4. Also report how many backward steps were taken and whether they stalled.
- Run the same check at one point near a transition (θ ≈ 1.87π), where CTM converges slowly, to see whether the backward series still converges.
- If the gate fails, try in order: tighter forward `corner_tol`, larger χ, more backward `max_sweeps`. If it still fails, switch to the commented-out linear-solve backward ("option 2" in `fixed_pt.py`) as a local patch, and record that in `yastn_findings.md`.

## 5b. Optimizer pitfalls **[impl]**

- `torch.optim.LBFGS(max_iter=1)` defaults to `max_eval = 1`, which caps the strong-Wolfe line search at `max_eval - 1 = 0` extra evaluations: whenever the full step t = 1 goes uphill it returns t = 0 and the chain stalls with an unchanged energy. `Chain` sets `max_eval = max_ls + 1` (21).
- torch's own stopping rules (`tolerance_grad` 1e-7, `tolerance_change` 1e-9, both absolute) fire early: raw-parameter gradients are ~1e-5 near convergence. They are set to 0; `Chain.run` stops on |ΔE| < `tol_E` over `window` steps or |g| < `tol_grad`.
- torch evaluates the closure at the start of every `step()`, where the previous line search ended; `Chain` caches that evaluation, so a step costs the line-search evaluations only (often 1).
- The optimizer energy is measured with the truncated CTM of the optimization (`tol` 1e-8 keeps ~D² states). At D = 3 it was 9e-6 below the validated value; always report `validate`.

## 6. Failure handling

| Failure | Action (in order; give up after 3 attempts → mark chain `failed`) |
|---|---|
| `NoFixedPointError` code 1, forward not converged | cold restart (`init='dl'`) → max_sweeps ×2 → perturb tensors by 1e-3 noise and reset optimizer history |
| `NoFixedPointError`, sectors change (symmetric tensors) | fix the virtual charge sectors (`D_block`) for the chain; never let them float during AD |
| Degenerate singular values cut inside a multiplet (symmetric points such as θ=π/4, or FM/FQ) | multiplet-aware truncation: `truncation_mask_multiplets(S, keep_multiplets=True, D_total=chi, eps_multiplet=1e-8)` as in the upstream test |
| Line search fails / E increases | reset L-BFGS history; if it fails twice, take 20 Adam steps (lr 1e-3) then return to L-BFGS |
| NaN in gradient | restore last checkpoint, perturb, reset optimizer |

Every failure is logged with its type and the step number.

## 7. Seeds, D ladder, and phase selection

For each (model params, cell), seed chains from **all** of these:

1. **Product seeds** from the mean-field minimizer: AFM3 (optimized, reduced m, not 120° coherent), AFQ, FM, FQ, each padded to D with noise 1e-3.
2. **Simple-update state** at the same D (EnvBP, cheap; §4.1).
3. **Pad from D−1:** the best validated state at D−1, embedded in D with noise 1e-3 on the new bond indices.
4. **Hysteresis:** the validated state from the neighbouring θ (or J₂), in both scan directions.
5. **Random:** 2 seeds.

D ladder: D = 2, 3, 4, … up to the cost limit. At each D, the ground state is the lowest **validated** energy over chains and unit cells. Keep all chains. The energy-vs-θ curves of each phase-seeded branch give θ_c by crossing; this is the first-order case.

Near a possibly continuous transition (§8 of SPEC; 120°→AFQ with J₂): crossing curves are not enough. Track m(D) and |Q|(D) and the correlation length ξ(D) per branch.

## 8. Checkpointing and resume

Save the following every step, atomically (write to a temp file, then rename):
- the raw (unnormalized) leaf tensors that the optimizer updates. The loss normalizes them (§3), so their norms drift, and the normalized `psi` alone cannot restore the optimizer's coordinates;
- `bbh_tri.yastn_compat.peps_to_dict(psi)` (not `psi.to_dict()`, which loses non-default TriangularLattice geometry);
- the optimizer state (L-BFGS history refers to the raw leaf tensors);
- the last converged env (`env.to_dict()`);
- the RNG state;
- the step log;
- metadata (config, git hash, yastn version, torch version).

Resume picks up the latest checkpoint and reuses the env as a warm start. One HDF5 group per chain.

## 9. Runtime

- Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` per chain; parallelize over chains (SLURM array), not threads.
- GPU is optional: `fp_ctmrg(..., fwd_devices=...)` exists but is untested here.
- Profile one step (forward + backward) at D = 2, 3, 4 with χ = 2D² before sizing the array jobs. The fullrank fixed-point SVD is the expected bottleneck at large χ.
- **[impl] Memory (unrolled AD stores every CTM sweep).** Dense D = 4: ~88 GB on the CPU at χ = 48; on an 80 GB A100 χ = 48 fits and χ = 64 does not (~76 GB per evaluation; trial points OOM). Dense D = 5, χ = 75: ~5 GB per sweep, ~200 GB at a typical ~36 sweeps, 299 GB peak seen; budget ≥ 320 GB at `max_sweeps` 50.

## 10. First experiment (proposed acceptance for this workflow)

At θ=π/4, √3×√3 cell, dense:

1. The gradient gate (§5) passes at D=2.
2. D=2 AD, seeded from the D=2 EnvBP state (1.8795, anisotropic):
   - the validated energy is below 1.8795;
   - report whether C₃ is restored.
3. D=3 AD, seeded from the D=3 EnvBP state:
   - the validated energy is ≤ 1.68856 (the EnvBP value);
   - it converges in χ.
4. Restart the D=3 chain from a checkpoint and get identical subsequent steps (deterministic resume).

These go into `docs/yastn_findings.md` and the M4 note.

**[impl] Results (2026-10-18), seeds from EnvBP runs started at random product states:**
1. Gradient gate: see §5.
2. D = 2 (EnvBP 1.8401083, χ_opt 8): AD converges to 1.8401000 in 7 steps (a local minimum next to the BP state; the anisotropic 1.8795 state of the sanity run was not used).
3. D = 3 (EnvBP 1.68853 at χ = 18, `tol` 1e-8; 1.68856 in the sanity run), χ_opt = 18: converged in 26 L-BFGS steps (~2 s each). Optimizer energy 1.6740229; validated 1.6740229 / 1.6740321 / 1.6740320 at χ = 9 / 18 / 27, i.e. **1.67403**, 0.0145 below EnvBP and converged in χ. C₃: h, v = 0.55742, d = 0.55920 (spread 1.8e-3). m = 0.47, 0.60, 0.47 and |Q| = 0.59, 0.46, 0.59 per sublattice.
4. Resume: `test_chain_lowers_energy_and_resumes_deterministically` (D = 2) gives bitwise identical steps after reloading a checkpoint.
