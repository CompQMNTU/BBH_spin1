# YASTN findings (M0)

yastn `1.6.2.dev320+g7ed8ad0e6`, built with
`uv run --no-project --with "yastn @ git+https://github.com/yastn/yastn@7ed8ad0e6" --with numpy,scipy`.
Line numbers refer to that version. Checked 2026-10-14; re-check if the pin in `pyproject.toml` changes.
The test scripts were throwaway session scripts. Port the relevant ones into `tests/` (`test_routing.py`, `test_product_states.py`) at M0/M1.

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

## U4 — `fp_ctmrg` on TriangularLattice (code read only)

Uses only `sites()`, `site2index`, `nn_site`, `Nx`, `Ny` (`fixed_pt.py:577-610`), so it looks geometry-agnostic. Not run; needs a torch test at M4.

## Symmetric configs

A U1-symmetric `Spin1` on TriangularLattice with path gates, NTU at D=3, and then CTM runs without errors.

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

Reference: Niesen–Corboz D→∞ 1.630(1); best finite-D 1.632 (D=12, SU(3)-symmetric).

Notes:
- The ED row is an operator check only. With 16 bonds on 9 sites it is not comparable per site with the infinite lattice.
- D=3 restores C₃ (h, v, d within 0.4%). D=2 does not: the diagonal bond is much higher and ⟨Sz⟩ = ±0.76–0.81, so the state is likely stuck in an anisotropic configuration. Not investigated; treat D=2 as unreliable.
- CTM did not reach `corner_tol=1e-11` in 300 sweeps on the final runs (remaining change 1e-7 to 1e-9). At D=2, χ=6 reported max_D=4.

## Performance

- Default threading on 48 cores: D=2 NTU took ~10 s per step. Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Single-threaded, D=3 NTU did not finish 100 steps in 8 min. Profile NTU before M3; EnvBP is the practical small-D default.
