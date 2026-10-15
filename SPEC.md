# SPEC — Spin-1 BBH model on the triangular lattice with YASTN iPEPS (+ NNN extension)

Reference paper: I. Niesen & P. Corboz, *A ground state study of the spin-1 bilinear-biquadratic Heisenberg model on the triangular lattice using tensor networks*, PRB **97**, 245146 (2018), arXiv:1805.00354.
Library: YASTN (`yastn.tn.fpeps`), https://github.com/yastn/yastn, docs https://yastn.github.io/yastn/

Owner: Ying-Jer Kao (NTU). This file is the contract for Claude Code: read it fully before writing code; do not change conventions in §2 without updating this file.

---

## 0. Goals

1. **Reproduce** the NN phase diagram of Niesen–Corboz (four phases AFM3 / AFQ / FM / FQ, all first order) with YASTN, to the precision of the benchmarks in §3.
2. **Extend** to second-neighbor (triangular-lattice NNN, distance √3) couplings:
   H = Σ_⟨ij⟩ a_ij [cos θ S_i·S_j + sin θ (S_i·S_j)²] + Σ_⟨⟨ij⟩⟩ a_ij [J₂ S_i·S_j + K₂ (S_i·S_j)²]
   with the spatial anisotropy of the paper (Sec. IV F, Fig. 12) as an option: a_ij = 1 on the horizontal chains (class `h`) and a_ij = J_anis on all bonds between chains. The paper (NN only) scales the vertical and diagonal bonds, 0 ≤ J_anis ≤ 1, J_anis = 0 = decoupled chains, J_anis = 1 = isotropic; it used this to bound the Haldane phase (SU, D = 9, −π/4 < θ < π/4). Every NNN bond also connects different chains, so here J_anis scales them too (our choice; keeps J_anis = 0 decoupled). Default J_anis = 1.
3. Build it as a small, tested package so parameter scans can run unattended on the cluster.

Non-goals (for now): finite-T, excitations, SU(3)-symmetric tensors (YASTN is abelian-only).

### Terminology warning (important)
In the paper, "next-nearest neighbor" refers to the **square-lattice diagonal** that encodes the triangular NN bond. In this project, **NNN always means triangular-lattice second neighbors (|r| = √3)**. These are *not* the same bonds.

---

## 1. What YASTN already provides (checked against source + tests, yastn 1.6.2.dev320+g7ed8ad0e6, 2026-10-14)

Details, line references and the test scripts' results: `docs/yastn_findings.md`.

| Need | YASTN object | Status |
|---|---|---|
| Spin-1 operators | `yastn.operators.Spin1(sym='dense'|'Z3'|'U1')`, basis (sz=+1,0,−1) for dense; `U1` charge = sz (U1 `to_numpy` orders charges ascending: −1,0,+1) | ✅ |
| Triangular geometry on square PEPS | `fpeps.TriangularLattice(dims=(3,3), boundary='infinite', full_patch=False)` — √3×√3, 3 tensors, `site2index = (y−x) mod 3`; `full_patch=True` → all Nx·Ny sites unique, any dims (tested (2,2),(2,3),(4,4),(6,6)); `bonds(dirn='h'|'v'|'d')` | ✅ infinite |
| — OBC | `boundary='obc'` is **broken** for triangular: with `full_patch=True`, `bonds('d')` contains `Bond(None, …)`; with `full_patch=False`, `dims` is ignored (3 sites, bonds to nonexistent `Site(0,3)`) | ❌ filter bonds ourselves |
| Diagonal bond convention | `Bond(s_b, s_r)` = `((x+1,y),(x,y+1))`, i.e. the **anti-diagonal** (bottom-left ↔ top-right) | ✅ |
| General unit cells | `fpeps.RectangularUnitcell(pattern)` — **no** diagonal bonds (`bonds('d')` silently returns h+v); for triangular use `TriangularLattice(full_patch=True, dims=…)` | ✅ |
| Gates | `fpeps.Gate(G, sites)`; `G` may be an MPO (or list of tensors) along a **path of NN sites**; a 2-site `G` with `sites` = full NN path gets identities on the middle sites (`fill_eye_in_gate`) | ✅ tested vs ED |
| Gate builders | `gates.gate_nn_exp(step, I, H, bond=None)`, `gates.gate_local_exp(step, I, H, site=None)`, `yastn.fkron`, `gates.distribute(geometry, gates_nn, gates_local, symmetrize=True)` — **do not** use `distribute` with `TriangularLattice` (it emits bare diagonal gates, which raise) | ✅ / ⚠ |
| Evolution | `fpeps.evolution_step_(env, gates, opts_svd, method='mpo'|'nn', …)` with `EnvBP` (≈ simple update; works although the docstring omits it), `EnvNTU`, `EnvCTM` (full update), `EnvApproximate` | ✅ |
| CTMRG | `fpeps.EnvCTM(psi, init, leg)`, `.iterate_(opts_svd, moves='hv', method='2x2', max_sweeps, corner_tol)` (default method `'2x2 corner'`; `'2x2'` also accepted) | ✅ |
| Measurements | `measure_1site`, `measure_nn` (square NN only; **raises** on the diagonal), `measure_2x2`, `measure_line` (h/v lines only), `measure_nsite_exact` (any rectangle; 2×3 knight window tested), `measure_2site` (long-range, boundary MPS), `transfer_matrix_spectrum` (numpy only). `measure_nsite_exact_oe` **does not exist**. | ✅ |
| — rdm helpers | `envs/rdm.py`: `measure_rdm_nn`, `measure_rdm_diag`, `measure_rdm_2x2` **crash on dense configs** (`IndexError` from `O0.n[0]`); calling `rdm1x2`, `rdm2x1`, `rdm2x2_anti_diagonal` directly works | ❌ wrappers / ✅ low-level |
| AD / variational | `yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg` (torch backend, implicit differentiation, gauge fixing, `NoFixedPointError`). Returns **only `env`** (docstring says a tuple); module imports torch at top level | ✅ (code read; not run) |
| Custom symmetry | no built-in Z2xZ2; a `sym_abelian` subclass passed as `make_config(sym=cls)` works (SVD tested). `Z2xU1` exists as a class but is not accepted as a string | ✅ |
| Save / load | `to_dict` / `from_dict` round-trip the default √3×√3 TriangularLattice PEPS (`save_to_dict` deprecated). **Non-default cells (`full_patch`, OBC, other `dims`) silently reload with the default geometry**; save with `bbh_tri.yastn_compat.peps_to_dict` | ⚠️ workaround |

**Defaults that iterate over all bonds break on TriangularLattice** (they hit the diagonal and raise `YastnError: Site(1, 0), Site(0, 1) are not nearest-neighbor sites.`): `measure_nn()`, `EnvBP.measure_nn()`, `truncate_(bond=None)`, `Peps.get_bond_dimensions()`, `gates.distribute`. Always pass explicit h/v bonds.

### M0 items (U1–U4)
- **U1 — resolved.** A bare 2-site diagonal gate raises the error above (EnvBP and EnvNTU). A 2-site gate given with `sites` = an NN path matched exact ED (overlap 1 to 1e-15) on a 2×3 OBC entangled state for both 3-site diagonal paths, the (1,1) path and all three 4-site knight paths, with EnvBP/EnvNTU in `mpo` and `nn` modes; wrong placements give overlap 0.86–0.91. `evolution_step_` truncates **every** bond along the path. Keep `test_routing.py` as a regression test.
- **U2 — resolved.** Diagonal bonds: use `measure_2x2`, `measure_nsite_exact`, or a direct `rdm2x2_anti_diagonal` contraction (all three agree to ~1e-15). Not `measure_nn`, `measure_rdm_diag` (dense crash), or `measure_line`.
- **U3 — partly.** `measure_nsite_exact` runs on a 2×3 window at small D. Memory/time at the target D, χ is still open; profile before M5.
- **U4 — resolved with a patch.** `fp_ctmrg` works on `TriangularLattice`. YASTN's SVD backward regularizes 1/σ and 1/(σᵢ±σⱼ) with an absolute 1e-12 (`safe_inverse` ignores its scaled ε), which corrupts gradients once CTM singular values or gaps fall below ~1e-6: 2% error at χ=8, 40% at χ=16. With `bbh_tri.yastn_compat.exact_svd_backward()` AD matches finite differences to ~1e-7 at χ = 4–16. All AD code must run forward and backward inside it. Numerically degenerate kept singular values (gap ≲ 1e-10·σ_max) still give unreliable gradients; the patch then raises `DegenerateSVDWarning`, which AD runs must treat as an error. Exactly zero kept singular values are undetectable, so AD runs must truncate with `opts_svd['tol'] > 0`. See `docs/yastn_findings.md` U4.

Record findings in `docs/yastn_findings.md`.

---

## 2. Conventions (fixed)

### 2.1 Coordinates
YASTN `Site(nx, ny)` = (row, column). Square-PEPS NN steps: r = (0,1), b = (1,0).
Triangular primitive vectors in (row, col) components: **a₁ = (0,1)**, **a₂ = (1,0)**, NN third bond **a₁ − a₂ = (−1,1)** (matches YASTN's diagonal). Angle(a₁,a₂) = 60°.

| shell | |r| | vectors (row, col) up to sign | square-PEPS footprint |
|---|---|---|---|
| NN (3) | 1 | (0,1), (1,0), (−1,1) | 1×2, 2×1, 2×2 |
| **NNN (3)** | √3 | **(1,1)**, **(−1,2)**, **(2,−1)** | **2×2**, **2×3**, **3×2** |
| 3rd (3) | 2 | (0,2), (2,0), (−2,2) | 1×3, 3×1, 3×3 |

Unit test: for each vector v = n₁a₁ + n₂a₂, |v|² = n₁² + n₂² + n₁n₂ must give 1, 3, 4.

Consequences:
- NNN connects **same** 3-sublattice sites ((y−x) mod 3 invariant): √3×√3 cell remains valid for AFM3/AFQ/FQ with J₂.
- Two of three NNN bonds are knight moves → 4-site paths for gates and 2×3 windows for measurement. This is the dominant cost of the extension. `evolution_step_` truncates all 3 bonds of a 4-site path per gate (§1 U1), so expect a larger truncation bias. M5(c) must quantify it, not just check total energy.

### 2.2 Operators
- S·S = Sz⊗Sz + ½(S⁺⊗S⁻ + S⁻⊗S⁺); (S·S)² as the matrix product of the 9×9 operator — build both with `yastn.fkron` so symmetric versions keep block structure. Add `0*fkron(I,I)` as the YASTN gate builders do.
- Quadrupole vector (S(S+1)=2), traceless components:
  Q = ( Sx²−Sy², (3Sz²−2)/√3, {Sx,Sy}, {Sy,Sz}, {Sz,Sx} ),  with (3Sz²−2)/√3 ≡ (2Sz²−Sx²−Sy²)/√3.
  (Earlier drafts had (2Sz²−2)/√3: not traceless, and it gives |Q|(|+1⟩)=0. Wrong.)
  Checks: Q_i·Q_j = 2(S_i·S_j)² + S_i·S_j − 8/3 exactly; |Q|(|0⟩) = 2/√3; |Q|(|±1⟩) = 1/√3.
  In a U1-symmetric state every Sz-charged component (S±², {S±,Sz}, hence Sx²−Sy², {Sx,Sy}, {Sy,Sz}, {Sz,Sx}) has zero expectation by symmetry. So |⟨Q⟩| = |⟨3Sz²−2⟩|/√3 there, and `measure.py` must use that branch for U1 states rather than attempt charged operators. Test: on a U1-compatible product state (|0⟩, |±1⟩) this equals the dense |⟨Q⟩|.
- Order parameters per site: m = |⟨S⟩|, |Q| = |⟨Q⟩| (paper's "Q-norm"); report sublattice-resolved values and the unit-cell average.
- Bond energies reported per bond **class** (h, v, d, nnn_d, nnn_k1, nnn_k2) — needed for anisotropy/nematic diagnostics.

### 2.3 Energies
Energy **per site** using form (1) of the paper (not the J_S/J_Q form, which differs by a constant). Per site there are 3 NN bonds and 3 NNN bonds.

---

## 3. Reference numbers / acceptance tests

From the paper (Niesen–Corboz 2018):
- θ = π/4 (SU(3) point): E/site(D→∞) = **1.630(1)**; best finite-D = 1.632 (variational, D=12, SU(3) symmetry exploited). Relation E_BBH = (E_SU(3)+3)/√2.
- θ = 0 (Heisenberg): E/site(D→∞) = **−1.8368(5)**; m stays finite.
- FQ→AFM3 transition: **θ_c = 1.873(7)π** (full update, D ≤ 11 for FQ with U1, D ≤ 9 for AFM3).
- Other transitions exactly at π/4, π/2, 5π/4; all first order.
- FM phase π/2 < θ < 5π/4 is an exact product state: **E/site = 3(cos θ + sin θ)**.
- Note: the paper quotes the mean-field θ_c as arctan(−2) ≈ 1.648π (Fig. 2) but writes 1.658π in Sec. IV F — **1.648π is correct** (checked numerically).

Independently computed product-state (D=1, 3-sublattice) values (my check, to be reproduced in tests):
| θ | E/site (D=1) | state |
|---|---|---|
| 0 | −1.5 | 120° |
| π/4 | 3/√2 = 2.1213 | degenerate manifold of mutually orthogonal triads (e.g. \|+1⟩,\|0⟩,\|−1⟩ or x,y,z directors). The 120° coherent state is **not** in it: it gives 2.254 |
| π/2 | 3.0 | AFQ = FM degenerate |
| 5π/4 | −3√2 = −4.2426 | FM = FQ degenerate |
| arctan(−2)+2π ≈ 1.64758π | −12/√5 = −5.36656 | FQ/AFM3 mean-field transition; FQ energy = 6 sin θ (tests use the exact angle, not 1.6476π) |

Product-state formulas: E_FQ = 6 sin θ, E_AFQ = 3 sin θ, E_FM = 3(cos θ + sin θ).
Q-norm of |sz=0⟩ = 2/√3; of |±1⟩ = 1/√3.

**Mean-field AFM3 ≠ 120° coherent state** (checked with a general 3-sublattice product-state minimizer, 2026-10-14):
- The 120° spin-coherent state has ⟨S·S⟩ = −1/2, ⟨(S·S)²⟩ = 1.5625, so E = 3(−½ cos θ + 1.5625 sin θ). This is optimal only at θ = 0.
- For 3π/2 < θ < 2π the optimal product state is a coplanar 3-sublattice state with **reduced** moment: m ≈ 0.47, 0.78, 0.95, 0.99 at θ = 1.66, 1.7, 1.8, 1.9 π. At 1.7π: E_opt = −4.9576 vs E_120coh = −4.674.
- The mean-field FQ→AFM3 transition at θ = arctan(−2)+2π = 1.6476π is **continuous**: m → 0, and E_opt − E_FQ ∝ (θ−θ_c)². It is not a level crossing.
- Consequence: `states.py` AFM3 seeds and M2 tests must use the optimized general product state (store the minimizer in tests/), not a 120° coherent state.

Exact-diagonalization check (to write): primary role is catching operator / bond-routing bugs, not accuracy.
- **Do not use a 3×3 torus when J₂, K₂ ≠ 0.** Mod 3, all three NNN vectors (1,1), (−1,2), (2,−1) reduce to ±(1,1). Each site then has only 2 distinct NNN partners, and 3rd neighbors coincide with NN. Bond classes become indistinguishable, which defeats the routing test.
- Primary: **OBC** 3×3 triangular cluster (3⁹ = 19683 states, `scipy.sparse.linalg.eigsh`) (6 h + 6 v + 4 diagonal = 16 NN bonds). It contains all bond classes, including both knight moves. `TriangularLattice(boundary='obc')` is broken (§1), so build the finite PEPS with `full_patch=True` and filter out bonds containing `None` in `lattice.py`; the ED and PEPS bond lists must come from the same function. Reference: E0 = 6.0015950 at θ=π/4, J₂=K₂=0 (docs/yastn_findings.md).
- Gate-level check (in `test_routing.py`): apply a single gate exp(−τ h) to a product/random finite PEPS without truncation, contract exactly, and compare with the ED vector. This isolates routing from optimization.
- Optional periodic check: the torus must keep the shells distinct. Among axis-aligned Lx×Ly tori, any side = 3 collapses NNN. **4×4 is the smallest** with distinct NN+NNN (3¹⁶ = 43,046,721 states; 5,196,627 ≈ 5.2·10⁶ in Sz=0). 5×5 is the smallest that also separates 3rd neighbors. 4×4 cannot host 3-sublattice order, which is fine for a routing check. Assert bond multiplicities explicitly (3N NN, 3N NNN).

---

## 4. Algorithms

### 4.1 Mapping paper methods → YASTN
| Paper | YASTN |
|---|---|
| Simple update (triangle variant) | `EnvBP` metric; triangle gate as 3-site MPO along path (x+1,y)→(x,y)→(x,y+1), `method='mpo'` |
| Full update | `EnvCTM` metric in `evolution_step_` (or `EnvNTU` as cheaper default) |
| Variational update | implicit-AD CTM: `fp_ctmrg` (torch), L-BFGS — workflow in `docs/ad_workflow.md` |
| Hysteresis / warm start | load state from neighboring θ, continue |
| U(1) for FQ/FM | `Spin1(sym='U1')` |

### 4.2 Gates
- **Option A (paper-like, preferred):** decompose H = Σ_▲ h_▲ over up-triangles; each up-triangle {(x,y),(x,y+1),(x+1,y)} contains one h, one v and one diagonal bond, and every NN bond belongs to exactly one up-triangle. Build exp(−τ h_▲) on 27-dim space, SVD into a 3-site MPO along the L-path (x+1,y)–(x,y)–(x,y+1), apply with `method='mpo'`. One gate application per triangle, followed by sequential truncation of both L-path bonds ((x+1,y)–(x,y) and (x,y)–(x,y+1)), per §1 U1.
- **Option B:** bond-by-bond: h, v via `gate_nn_exp`; diagonal via 2-site op on a 3-site path (identity on middle site) — only after U1 is resolved. Do not feed `TriangularLattice` straight into `gates.distribute`: it iterates `geometry.bonds()`, which includes the diagonal 2-site bonds, and those should raise (§1 U1). Build the gate list yourself.
- **NNN gates:**
  - (1,1): path (x,y)→(x,y+1)→(x+1,y+1) [or via (x+1,y)]; alternate the two paths between sweeps to avoid bias.
  - (−1,2), (2,−1): 4-site paths (Manhattan distance 3). Either straight-then-turn path; alternate. Expect higher truncation error — monitor `accumulated_truncation_error`.
- Second-order Trotter: `gates.distribute(..., symmetrize=True)` only appends the reversed list of the gates **it** built (`gates.py`, `distribute`), with each gate built at step τ/2. Triangle and NNN gates are built by `bbh_tri.gates`, so do the same there: build every gate as exp(−(τ/2)h) and append the reversed list (optionally with reversed `sites`). τ schedule e.g. 0.1 → 0.05 → 0.02 → 0.01, stop on energy change < 1e-7 per stage.

### 4.3 Environment / measurement
- CTMRG `method='2x2'`, χ ≥ D², `corner_tol=1e-10`; check χ convergence explicitly (χ, 1.5χ).
- NN: `measure_nn` for h/v (explicit bonds only), `measure_2x2` or `measure_nsite_exact` for the diagonal (U2).
- NNN (1,1): `measure_2x2`. Knight moves: `measure_nsite_exact` on 2×3 / 3×2 windows (cost check U3). For AD, energy must be differentiable through these.
- Long-range ⟨S_0·S_r⟩, ⟨Q_0·Q_r⟩ via `measure_2site` → structure factors at Γ, K, M.

### 4.4 Unit cells to scan
- √3×√3 (default `TriangularLattice()`): AFM3, AFQ, FQ, FM.
- `full_patch=True` with dims (2,2), (3,3), (4,4), (6,6) (paper went up to 6×6 — add section/figure reference). (2,2) is required for stripe / 4-sublattice orders expected at larger J₂; (6,6) as a catch-all.
- Commensurability: (2,2) and (4,4) **cannot** host 3-sublattice (AFM3/AFQ) order, and (3,3) cannot host stripe/4-sublattice order. Only (6,6) holds both. Energy comparisons across cells must account for this; a cell that frustrates the competing order is not evidence against it. (Classically the J₁–J₂ model turns into an incommensurate spiral for J₂ ≳ J₁, outside the planned J₂ ≤ 0.3 cut.)
- Ground state at given parameters = lowest energy over all unit cells and initializations (random + phase-seeded + hysteresis from neighbors).

### 4.5 Symmetry choice
Symmetries are not needed for correctness; they buy bond dimension (paper: FQ with U(1) to D=11 vs dense AFM3 to D=9; SU(3) benchmark to D=12–16). An imposed symmetry forbids its spontaneous breaking, so pick per phase:

| Phase / state | On-site symmetry | YASTN | Notes |
|---|---|---|---|
| FM (moment ∥ z) | U(1) (Sz) | `Spin1(sym='U1')` | exact product state; symmetry barely matters |
| FQ (director ∥ z, \|0⟩) | U(1) | `Spin1(sym='U1')` | as in the paper |
| AFQ (directors x, y, z) | **Z₂×Z₂** = {e^{iπSx}, e^{iπSy}, e^{iπSz}} (commute for integer spin) | custom: no built-in Z2xZ2 in yastn/sym (has Z2, Z3, U1, U1xU1, Z2xU1, U1xU1xZ2) → define one by subclassing `sym_abelian` (model: `sym_Z2xU1.py`); operators written in the \|x⟩,\|y⟩,\|z⟩ basis. Fallback: `make_config(sym='Z2')` = Sz parity only | \|x⟩,\|y⟩,\|z⟩ are joint eigenstates (charges (0,1),(1,0),(1,1) up to convention); \|x⟩,\|y⟩ mix Sz=±1; not U(1) |
| AFM3, 120° coplanar | none | `dense` | coplanar order breaks every U(1); no on-site Z_n leaves it invariant |
| SU(3) point θ=π/4 | U(1)×U(1) (flavor counts) | custom config | choose degenerate state with sublattices \|+1⟩,\|0⟩,\|−1⟩; only way to approach D≈12–16 |
| NNN: collinear stripe / up-up-down | U(1) | `Spin1(sym='U1')` | collinear along z |
| NNN: tetrahedral (4-sublattice, non-coplanar) | none | `dense` | needs a (2,2)-compatible cell (§4.4); reported by Szasz et al. (§7) |
| NNN: disordered / spin-liquid candidate | U(1) | `Spin1(sym='U1')` | largest payoff (large D needed) |

Correction to earlier drafts: YASTN's `Spin1(sym='Z3')` is Sz mod 3, a subgroup of U(1). It is therefore valid wherever U(1) is (FM, FQ, collinear states, the SU(3) point), but always strictly weaker than U(1) there and useless for AFQ/AFM3. Do not use it.

Rules:
- Build and validate everything **dense** through M3; add symmetry as a config switch at M4.
- Never compare **fixed-D** energies across symmetry choices; compare D→∞ extrapolations (as the paper did for θ_c) or equal-cost runs.
- No spatial symmetry is imposed (YASTN only has C4v, which the triangular embedding breaks). Monitor C₃ restoration through per-class bond energies (h, v, d) instead.
- Seed symmetric runs from product states; for random symmetric starts, choose a sensible virtual-charge sector distribution and let `D_total` truncation redistribute.
- Test: each symmetric implementation must reproduce dense energies for its compatible product state (D=1) and agree with dense at small D (D=2–3) when started from the same state.

---

## 5. Package layout

```
BBH_spin1/                # repo root; Python package name: bbh_tri
  pyproject.toml          # uv-managed; deps: yastn (pinned to a git commit), numpy, scipy, h5py, pyyaml; extra `ad` = torch
  uv.lock                 # committed
  CLAUDE.md               # short pointer to this SPEC + run/test commands
  SPEC.md
  src/bbh_tri/
    lattice.py            # bond classes, NN/NNN/3rd vectors, path builder, sublattice map
    operators.py          # Spin1 wrappers, S·S, (S·S)^2, Q-vector, per-bond Hamiltonians h(J,K)
    hamiltonian.py        # Model dataclass: theta, J2, K2, J_anis, J3 (future); bond-class -> coupling
    gates.py              # triangle MPO gate, nn/diag/nnn path gates, Trotter schedules
    states.py             # initial states: random, product AFM3/AFQ/FM/FQ/m=2/3, load-from-file
    evolve.py             # imaginary-time driver (BP/NTU/CTM metric), convergence, checkpoints
    variational.py        # AD driver with fp_ctmrg + L-BFGS, NoFixedPointError recovery
    measure.py            # energy per bond class, m, |Q|, sublattice data, correlations, S(q)
    extrapolate.py        # 1/D extrapolation per paper §III B (midpoint + half-difference error)
    io.py                 # HDF5/JSON: state (to_dict), env, metadata (git hash, yastn version, opts)
    cli.py                # `bbh-run config.yaml`, `bbh-scan`, `bbh-hysteresis`
  tests/
    test_lattice.py       # vector norms, sublattice invariance of NNN, path validity
    test_operators.py     # Casimir=2, (S·S) spectrum {-2,-1,1}, Q traceless, Q·Q = 2(S·S)²+S·S−8/3, |Q|(|0>)=2/sqrt3, |Q|(|±1>)=1/sqrt3, U1 vs dense agree
    test_gates.py         # MPO gate reconstructs exp(-tau h) to 1e-12 (dense contraction)
    test_product_states.py# D=1 energies of §3 table, via full CTM measurement
    test_ed_small.py      # OBC 3x3 cluster ED vs finite PEPS (slow marker); optional 4x4 torus (§3)
    test_routing.py       # U1/U2: diagonal & knight gates act on correct sites (2-site ED on a 2x3 OBC patch)
  configs/                # yaml for benchmarks and scans
  scripts/slurm/          # array-job templates
  docs/yastn_findings.md
```

Every saved result records: model params, unit cell, D, χ, τ schedule, symmetry, initialization, energy per bond class, m, |Q|, truncation error stats, CTM convergence info, wall time, yastn version, git hash.

---

## 6. Milestones (each ends with passing tests + a short note in `docs/`)

**M0 — Recon (½ day).** Set up the uv project and add YASTN pinned to a git commit (`uv add git+https://github.com/yastn/yastn@<commit>`); read `_evolution.py`, `_gates_auxiliary.py`, `envs/_env_ctm.py`; resolve U1–U4; write `docs/yastn_findings.md`. *Accept:* routing test passes or a documented workaround exists.

**M1 — Operators & lattice.** `lattice.py`, `operators.py`, `hamiltonian.py` + tests. *Accept:* all unit tests in test_lattice/test_operators pass in dense and U1.

**M2 — D=1 / product states.** Measure product states through the full YASTN pipeline. *Accept:* §3 product-state table reproduced to 1e-10 using the optimized general product state (not 120° coherent) for AFM3; mean-field onset of m (continuous) at arctan(−2)+2π, with E = 6 sin θ for 3π/2 < θ ≤ θ_c^MF.

**M3 — NN imaginary-time evolution.** Triangle-gate simple update (EnvBP) and NTU; √3×√3 cell. *Accept:* (a) FM region energies = 3(cosθ+sinθ); (b) θ=0 at D=4–6 gives E below the D=1 value and approaches −1.84 trend; (c) 80-point θ scan at D≤6 shows the four phases with jumps in m and |Q| at π/4, π/2, 5π/4, and near the FQ/AFM3 boundary (paper: 1.873(7)π; small-D simple update may shift it) (paper Fig. 5 qualitatively).

**M4 — Full update + AD, symmetries, benchmarks.** EnvCTM-metric FU and `fp_ctmrg` variational optimization following `docs/ad_workflow.md` (its gradient gate §5 must pass first); add the symmetry switch of §4.5 (U1, custom Z2×Z2, U1×U1) with its dense-agreement tests. *Accept:* θ=π/4 energy within 0.005 of 1.632 **with the U1×U1 config** (the paper needed SU(3) symmetry and D=12; dense alone is not expected to reach this — record the dense value as a separate data point); θ=0 extrapolation consistent with −1.8368(5) within combined error; FQ (U1) vs AFM3 (dense) hysteresis at θ = 1.86–1.89π gives θ_c within 1.873(7)π.

**M5 — NNN extension.** Add J₂, K₂ gates and measurements. *Accept:* (a) routing tests for (1,1), (−1,2), (2,−1) pass against ED on a small OBC patch; (b) D=1 energies with J₂ match an independent product-state minimizer (NNN contributes Σ_sublattices h₂(A,A) per site in the √3×√3 cell); (c) energy invariant under the two choices of gate path at convergence (within truncation error).

**M6 — Physics scans.** Start with K₂ = 0 and J₂ ≥ 0 (antiferro) along cuts through the NN phases. Priority cuts:
1. θ = 0 (Heisenberg) vs J₂/|J₁| ∈ [0, 0.3]: 120° → stripe (classical threshold J₂ = 1/8); check for intermediate phases rather than assume one.
2. Near the FQ/AFM3 boundary (θ ≈ 1.85–1.9π) vs J₂: does NNN open an intermediate phase?
3. AFQ region (π/4 < θ < π/2) and its boundary with 120° vs J₂: test Szasz et al.'s direct continuous 120°→AFQ transition (§7); first-order hysteresis tools alone are not enough here — also track how m and |Q| vanish with D.
Use unit cells §4.4, D up to the cost limit, compare energies across cells, and look at bond-energy anisotropy (nematic / VBS signatures) and S(q).

---

## 7. Prior work to compare against (NNN)

- A. Szasz, C. Wang, Y.-C. He, arXiv:2206.04087 — DMRG on the spin-1 triangular model with **NN + NNN bilinear** and **NN biquadratic** couplings (i.e. J₂ only, no K₂). Per their abstract (opened 2026-10-14): three magnetically ordered phases (120°, stripe, **tetrahedral**), two nematic phases (FQ, AFQ), and a likely **direct continuous 120° → AFQ transition rather than a spin liquid** in between. (Earlier drafts of this SPEC wrongly said they found a disordered/spin-liquid phase.) This is the natural first target: the open questions for iPEPS (2D, no cylinder bias) are whether the 120°–AFQ transition is really direct and continuous, and where the stripe/tetrahedral boundaries lie. Before mapping our (θ, J₂) to their couplings, read their Hamiltonian convention from the paper body (biquadratic as K(S·S)² vs a Q·Q form shifts the mapping); J = cos θ, K = sin θ is only a guess until checked. Unverified (from an earlier draft): that the J–K–J₂ model was previously treated only in a large-S expansion.
- Before M6, do a quick literature pass for later work (2022–2026) on the same model and on J₃ variants relevant to NiGa₂S₄; add to `docs/literature.md`. Do not cite anything not actually opened.

---

## 8. Pitfalls / guardrails

- All NN-model transitions are first order (paper). With NNN this is **not** guaranteed: Szasz et al. find a likely continuous 120°→AFQ transition, and the mean-field FQ→AFM3 onset is continuous (§3). Regardless of order: never infer a phase boundary from a single initialization; always compare energies of distinct seeded states (hysteresis), as in the paper.
- Near FM, states are ~product: D→∞ extrapolation is unnecessary and fits will be noise.
- AFM3 runs can drift into FQ at high D near θ_c (paper footnote 75); flag runs whose m drops discontinuously during optimization.
- Simple update can miss phases near transitions; never report SU-only boundaries without FU/AD checks.
- Knight-move NNN terms make every evolution step and energy evaluation markedly more expensive; profile at D=3 before launching scans.
- Threading: set `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) for small-D runs and in every SLURM template. With default threading on 48 cores, D=2 NTU took ~10 s/step. Even single-threaded, D=3 NTU did not finish 100 steps in 8 min at θ=π/4, so NTU cost must be profiled before M3 scans; EnvBP is the practical default at small D.
- For AD: `fp_ctmrg` returns only `env`, whatever its docstring says. Wrap it in `NoFixedPointError` handling (perturb and retry), and checkpoint every optimizer step.
- `transfer_matrix_spectrum` is numpy-only; compute correlation lengths in a separate post-processing pass.
- Keep numpy backend for evolution, torch only for AD; never mix configs in one state object.

---

## 9. CLAUDE.md

The agent instructions (commands, working rules, YASTN pitfalls) live in `CLAUDE.md` at the repo root. Keep it in sync with §1, §2, §6 and §8 when those change.
