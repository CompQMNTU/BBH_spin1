# Ground-state search by implicit-AD CTM (design)

Status: design for M4 (`src/bbh_tri/variational.py`). Nothing here is implemented or tested yet.
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
ctm_opt:  {chi: 2D², method: 2x2, corner_tol: 1.0e-8, max_sweeps: 200,
           svd_policy: fullrank, multiplets: true, init: warm|dl|eye}
optim:    {method: lbfgs, max_steps: 300, history: 20, line_search: strong_wolfe,
           tol_E: 1.0e-8, tol_E_window: 10, tol_grad: 1.0e-6}
validate: {chis: [2D², 3D², 4D²], corner_tol: 1.0e-10, max_sweeps: 500}
runtime:  {threads: 1, device: cpu, checkpoint_every: 1}
```

χ_opt = 2D² is a starting point. The validation stage (§4) decides whether that was enough.

## 3. Inner loop (closure evaluated by the optimizer)

Parameters are the raw data `t._data` of the unique site tensors (3 for √3×√3), with `requires_grad=True`. This is the same pattern as the upstream test.

1. **Build the state.** Copy the template tensors, insert the parameters, and normalize each tensor by its norm (inside the graph). Build `Peps(geometry, tensors)`.
2. **Environment.**
   - `warm` (default after step 0): start from the previous step's converged env with `.detach()` and the new `psi` attached.
   - `dl` / `eye`: cold start, used on retry.
   - Warm start is an untested speed optimization. If it causes trouble, fall back to `dl`, as the upstream test does.
3. **Fixed point.**
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
   - E(χ_val) is not more than 1e-5 **below** E(χ_opt). A larger drop means the optimizer exploited a too-small environment (χ bias); rerun with larger χ_opt.
   - CTM converged at every χ_val.

## 5. Gradient quality gate (before any production chain)

**Status (M0, 2026-10-15): fails.** AD vs FD differ by ~2% at χ=8 and ~40% at χ≥16 at D=2. The error is in differentiating the CTM step, not in the fixed-point series (`docs/yastn_findings.md` U4). Resolve this first.

- **Finite-difference check** at θ=π/4, D=2, χ=8, √3×√3, dense float64:
  - Use a random direction v and compare (E(x+εv) − E(x−εv))/2ε with ∇E·v for ε ∈ {1e-3, 1e-4, 1e-5}.
  - Acceptance: relative error < 1e-4. Also report how many backward steps were taken and whether they stalled.
- Run the same check at one point near a transition (θ ≈ 1.87π), where CTM converges slowly, to see whether the backward series still converges.
- If the gate fails, try in order: tighter forward `corner_tol`, larger χ, more backward `max_sweeps`. If it still fails, switch to the commented-out linear-solve backward ("option 2" in `fixed_pt.py`) as a local patch, and record that in `yastn_findings.md`.

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
- `psi.to_dict()`;
- the optimizer state;
- the last converged env (`env.to_dict()`);
- the RNG state;
- the step log;
- metadata (config, git hash, yastn version, torch version).

Resume picks up the latest checkpoint and reuses the env as a warm start. One HDF5 group per chain.

## 9. Runtime

- Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` per chain; parallelize over chains (SLURM array), not threads.
- GPU is optional: `fp_ctmrg(..., fwd_devices=...)` exists but is untested here.
- Profile one step (forward + backward) at D = 2, 3, 4 with χ = 2D² before sizing the array jobs. The fullrank fixed-point SVD is the expected bottleneck at large χ.

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
