# SPEC — Spin-1 BBH model on the triangular lattice with YASTN iPEPS (+ NNN extension)

Reference paper: I. Niesen & P. Corboz, *A ground state study of the spin-1 bilinear-biquadratic Heisenberg model on the triangular lattice using tensor networks*, PRB **97**, 245146 (2018), arXiv:1805.00354.
Library: YASTN (`yastn.tn.fpeps`), https://github.com/yastn/yastn, docs https://yastn.github.io/yastn/

Owner: Ying-Jer Kao (NTU). This file is the contract for Claude Code: read it fully before writing code; do not change conventions in §2 without updating this file.

---

## 0. Goals

1. **Reproduce** the NN phase diagram of Niesen–Corboz (four phases AFM3 / AFQ / FM / FQ, all first order) with YASTN, to the precision of the benchmarks in §3.
2. **Extend** to second-neighbor (triangular-lattice NNN, distance √3) couplings:
   H = Σ_⟨ij⟩ [cos θ S_i·S_j + sin θ (S_i·S_j)²] + Σ_⟨⟨ij⟩⟩ [J₂ S_i·S_j + K₂ (S_i·S_j)²]
   with the anisotropy knob J_anis of the paper kept as an option.
3. Build it as a small, tested package so parameter scans can run unattended on the cluster.

Non-goals (for now): finite-T, excitations, SU(3)-symmetric tensors (YASTN is abelian-only).

### Terminology warning (important)
In the paper, "next-nearest neighbor" refers to the **square-lattice diagonal** that encodes the triangular NN bond. In this project, **NNN always means triangular-lattice second neighbors (|r| = √3)**. These are *not* the same bonds.

---

## 1. What YASTN already provides (verified against docs, Oct 2026)

| Need | YASTN object | Status |
|---|---|---|
| Spin-1 operators | `yastn.operators.Spin1(sym='dense'|'Z3'|'U1')`, basis (sz=+1,0,−1) for dense; `U1` charge = sz | ✅ |
| Triangular geometry on square PEPS | `fpeps.TriangularLattice(dims=(3,3), boundary='infinite', full_patch=False)` — √3×√3, 3 tensors, `site2index = (y−x) mod 3`; `full_patch=True` → all Nx·Ny sites unique; `bonds(dirn='h'|'v'|'d')` | ✅ |
| Diagonal bond convention | `Bond(s_b, s_r)` = `((x+1,y),(x,y+1))`, i.e. the **anti-diagonal** (bottom-left ↔ top-right) | ✅ (from `_geometry.py`) |
| General unit cells | `fpeps.RectangularUnitcell(pattern)` — but it has **no** diagonal bonds; for triangular use `TriangularLattice(full_patch=True, dims=…)` | ✅ |
| Gates | `fpeps.Gate(G, sites)`; `G` may be an MPO (or list of tensors) along a **path of NN sites**; a 2-site op beyond NN can be given with `sites` = full path | ✅ |
| Gate builders | `gates.gate_nn_exp(step, I, H)`, `gates.gate_local_exp`, `yastn.fkron`, `gates.distribute(geometry, gates_nn, gates_local, symmetrize=True)` | ✅ |
| Evolution | `fpeps.evolution_step_(env, gates, opts_svd, method='mpo', …)` with `EnvBP` (≈ simple update), `EnvNTU`, `EnvCTM` (full update), `EnvApproximate` | ✅ |
| CTMRG | `fpeps.EnvCTM(psi, init, leg)`, `.iterate_(opts_svd, moves='hv', method='2x2', max_sweeps, corner_tol)` | ✅ |
| Measurements | `measure_1site`, `measure_nn` (square NN only), `measure_2x2`, `envs/rdm.py`: `measure_rdm_1site`, `measure_rdm_nn`, `measure_rdm_diag`, `measure_rdm_2x2`, `measure_line`, `measure_nsite_exact` (exact rectangular window), `measure_nsite_exact_oe`, `measure_2site` (long-range, boundary MPS), `transfer_matrix_spectrum` (numpy only) | ✅ |
| AD / variational | `yastn.tn.fpeps.envs.fixed_pt.fp_ctmrg` (torch backend, implicit differentiation, gauge fixing, `NoFixedPointError`) | ✅ |

### UNVERIFIED — must be checked in M0 before anything else
Source reading below was done against yastn `1.6.2.dev320+g7ed8ad0e6` (2026-10-14); re-check if the pinned version changes.
- **U1.** `gates.distribute` assigns the TriangularLattice diagonal `Bond((x+1,y),(x,y+1))` as a 2-site `sites` tuple. Those sites are *not* square NN.
  *Source reading:* `Peps.apply_gate_` (`_peps.py`) calls `nn_bond_dirn(s0, s1)` for each consecutive pair in `gate.sites`, and `nn_bond_dirn` (`_geometry.py`) raises `YastnError` for non-NN pairs → a bare diagonal 2-site gate should **raise**, not be auto-routed. A 2-site `G` with `sites` = a longer NN path gets identities inserted on the middle sites by `fill_eye_in_gate` (`_gates_auxiliary.py`), and `evolution_step_` then truncates **every** bond along the path in turn. This is the mechanism for Option B and the NNN gates (§4.2).
  *Still to do:* confirm both behaviors with a test (`test_routing.py`) and an ED comparison. **Do not trust any gate until that passes.**
- **U2.** Whether `measure_nn` accepts diagonal bonds of `TriangularLattice`. Assume **no**. Check `measure_rdm_diag` (`envs/rdm.py`) as a cheaper alternative to `measure_2x2`.
- **U3.** Whether `measure_nsite_exact` handles a 2×3 / 3×2 window at the target D, χ within memory (needed for NNN knight-move bonds).
- **U4.** Whether `fp_ctmrg` works with `TriangularLattice` geometry (it should — it only sees the square network — but test).

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
| 1.6476π | −5.3664 | FQ/AFM3 mean-field transition; FQ energy = 6 sin θ |

Product-state formulas: E_FQ = 6 sin θ, E_AFQ = 3 sin θ, E_FM = 3(cos θ + sin θ).
Q-norm of |sz=0⟩ = 2/√3; of |±1⟩ = 1/√3.

**Mean-field AFM3 ≠ 120° coherent state** (checked with a general 3-sublattice product-state minimizer, 2026-10-14):
- The 120° spin-coherent state has ⟨S·S⟩ = −1/2, ⟨(S·S)²⟩ = 1.5625, so E = 3(−½ cos θ + 1.5625 sin θ). This is optimal only at θ = 0.
- For 3π/2 < θ < 2π the optimal product state is a coplanar 3-sublattice state with **reduced** moment: m ≈ 0.47, 0.78, 0.95, 0.99 at θ = 1.66, 1.7, 1.8, 1.9 π. At 1.7π: E_opt = −4.9576 vs E_120coh = −4.674.
- The mean-field FQ→AFM3 transition at θ = arctan(−2)+2π = 1.6476π is **continuous**: m → 0, and E_opt − E_FQ ∝ (θ−θ_c)². It is not a level crossing.
- Consequence: `states.py` AFM3 seeds and M2 tests must use the optimized general product state (store the minimizer in tests/), not a 120° coherent state.

Exact-diagonalization check (to write): primary role is catching operator / bond-routing bugs, not accuracy.
- **Do not use a 3×3 torus when J₂, K₂ ≠ 0.** Mod 3, all three NNN vectors (1,1), (−1,2), (2,−1) reduce to ±(1,1). Each site then has only 2 distinct NNN partners, and 3rd neighbors coincide with NN. Bond classes become indistinguishable, which defeats the routing test.
- Primary: **OBC** 3×3 triangular cluster (3⁹ = 19683 states, `scipy.sparse.linalg.eigsh`) with the same bond list as `TriangularLattice(dims=(3,3), boundary='obc')`. It contains all bond classes, including both knight moves. Compare against the finite PEPS.
- Gate-level check (in `test_routing.py`): apply a single gate exp(−τ h) to a product/random finite PEPS without truncation, contract exactly, and compare with the ED vector. This isolates routing from optimization.
- Optional periodic check: the torus must keep the shells distinct. Among axis-aligned Lx×Ly tori, any side = 3 collapses NNN. **4×4 is the smallest** with distinct NN+NNN (3¹⁶ states, ≈1.1·10⁷ in Sz=0). 5×5 is the smallest that also separates 3rd neighbors. 4×4 cannot host 3-sublattice order, which is fine for a routing check. Assert bond multiplicities explicitly (3N NN, 3N NNN).

---

## 4. Algorithms

### 4.1 Mapping paper methods → YASTN
| Paper | YASTN |
|---|---|
| Simple update (triangle variant) | `EnvBP` metric; triangle gate as 3-site MPO along path (x+1,y)→(x,y)→(x,y+1), `method='mpo'` |
| Full update | `EnvCTM` metric in `evolution_step_` (or `EnvNTU` as cheaper default) |
| Variational update | AD with `fp_ctmrg` (torch), L-BFGS |
| Hysteresis / warm start | load state from neighboring θ, continue |
| U(1) for FQ/FM | `Spin1(sym='U1')` |

### 4.2 Gates
- **Option A (paper-like, preferred):** decompose H = Σ_▲ h_▲ over up-triangles; each up-triangle {(x,y),(x,y+1),(x+1,y)} contains one h, one v and one diagonal bond, and every NN bond belongs to exactly one up-triangle. Build exp(−τ h_▲) on 27-dim space, SVD into a 3-site MPO along the L-path (x+1,y)–(x,y)–(x,y+1), apply with `method='mpo'`, truncate once per triangle.
- **Option B:** bond-by-bond: h, v via `gate_nn_exp`; diagonal via 2-site op on a 3-site path (identity on middle site) — only after U1 is resolved.
- **NNN gates:**
  - (1,1): path (x,y)→(x,y+1)→(x+1,y+1) [or via (x+1,y)]; alternate the two paths between sweeps to avoid bias.
  - (−1,2), (2,−1): 4-site paths (Manhattan distance 3). Either straight-then-turn path; alternate. Expect higher truncation error — monitor `accumulated_truncation_error`.
- Second-order Trotter (`symmetrize=True`), τ schedule e.g. 0.1 → 0.05 → 0.02 → 0.01, stop on energy change < 1e-7 per stage.

### 4.3 Environment / measurement
- CTMRG `method='2x2'`, χ ≥ D², `corner_tol=1e-10`; check χ convergence explicitly (χ, 1.5χ).
- NN: `measure_nn` for h/v, `measure_2x2` for diagonal.
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
| AFQ (directors x, y, z) | **Z₂×Z₂** = {e^{iπSx}, e^{iπSy}, e^{iπSz}} (commute for integer spin) | custom: no built-in Z2xZ2 in yastn/sym (has Z2, Z3, U1, U1xU1, Z2xU1, U1xU1xZ2) → define one by subclassing `sym_abelian` (model: `sym_Z2xU1.py`); fallback `make_config(sym='Z2')` = Sz parity only + own operators in the \|x⟩,\|y⟩,\|z⟩ basis | \|x⟩,\|y⟩,\|z⟩ are joint eigenstates (charges (0,1),(1,0),(1,1) up to convention); \|x⟩,\|y⟩ mix Sz=±1; not U(1) |
| AFM3, 120° coplanar | none | `dense` | coplanar order breaks every U(1); no on-site Z_n leaves it invariant |
| SU(3) point θ=π/4 | U(1)×U(1) (flavor counts) | custom config | choose degenerate state with sublattices \|+1⟩,\|0⟩,\|−1⟩; only way to approach D≈12–16 |
| NNN: collinear stripe / up-up-down | U(1) | `Spin1(sym='U1')` | collinear along z |
| NNN: disordered / spin-liquid candidate | U(1) | `Spin1(sym='U1')` | largest payoff (large D needed) |

Correction to earlier drafts: YASTN's `Spin1(sym='Z3')` is Sz mod 3 and fits none of these phases except trivially as a subgroup of U(1) at the SU(3) point. Do not use it.

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
    test_operators.py     # Casimir=2, (S·S) spectrum {-2,-1,1}, Q-norm of |0>=2/sqrt3, U1 vs dense agree
    test_gates.py         # MPO gate reconstructs exp(-tau h) to 1e-12 (dense contraction)
    test_product_states.py# D=1 energies of §3 table, via full CTM measurement
    test_ed_small.py      # 3x3 cluster ED vs finite PEPS (slow marker)
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

**M3 — NN imaginary-time evolution.** Triangle-gate simple update (EnvBP) and NTU; √3×√3 cell. *Accept:* (a) FM region energies = 3(cosθ+sinθ); (b) θ=0 at D=4–6 gives E below the D=1 value and approaches −1.84 trend; (c) 80-point θ scan at D≤6 shows the four phases with jumps in m and |Q| at π/4, π/2, 5π/4, ~1.9π (paper Fig. 5 qualitatively).

**M4 — Full update + AD, symmetries, benchmarks.** EnvCTM-metric FU and `fp_ctmrg` variational optimization; add the symmetry switch of §4.5 (U1, custom Z2×Z2, U1×U1) with its dense-agreement tests. *Accept:* θ=π/4 energy within 0.005 of 1.632 **with the U1×U1 config** (the paper needed SU(3) symmetry and D=12; dense alone is not expected to reach this — record the dense value as a separate data point); θ=0 extrapolation consistent with −1.8368(5) within combined error; FQ (U1) vs AFM3 (dense) hysteresis at θ = 1.86–1.89π gives θ_c within 1.873(7)π.

**M5 — NNN extension.** Add J₂, K₂ gates and measurements. *Accept:* (a) routing tests for (1,1), (−1,2), (2,−1) pass against ED on a small OBC patch; (b) D=1 energies with J₂ match an independent product-state minimizer (NNN contributes Σ_sublattices h₂(A,A) per site in the √3×√3 cell); (c) energy invariant under the two choices of gate path at convergence (within truncation error).

**M6 — Physics scans.** Start with K₂ = 0 and J₂ ≥ 0 (antiferro) along cuts through the NN phases. Priority cuts:
1. θ = 0 (Heisenberg) vs J₂/|J₁| ∈ [0, 0.3]: 120° → (possible disordered) → stripe.
2. Near the FQ/AFM3 boundary (θ ≈ 1.85–1.9π) vs J₂: does NNN open an intermediate phase?
3. AFQ region (π/4 < θ < π/2) vs J₂.
Use unit cells §4.4, D up to the cost limit, compare energies across cells, and look at bond-energy anisotropy (nematic / VBS signatures) and S(q).

---

## 7. Prior work to compare against (NNN)

- A. Szasz, C. Wang, Y.-C. He, arXiv:2206.04087 — DMRG on the spin-1 triangular model with **NN + NNN bilinear** and **NN biquadratic** couplings (i.e. J₂ only, no K₂). They report several magnetic and nematic phases and at least one disordered phase that may be a spin liquid. This is the natural first target: map our (θ, J₂) to their (J, K, J₂) with J = cos θ, K = sin θ, and check whether iPEPS (2D, no cylinder bias) confirms the disordered region. Their paper also notes that the J–K–J₂ triangular model had previously been treated only in a large-S expansion.
- Before M6, do a quick literature pass for later work (2022–2026) on the same model and on J₃ variants relevant to NiGa₂S₄; add to `docs/literature.md`. Do not cite anything not actually opened.

---

## 8. Pitfalls / guardrails

- First-order transitions everywhere: never infer a phase boundary from a single initialization; always compare energies of distinct seeded states (hysteresis), as in the paper.
- Near FM, states are ~product: D→∞ extrapolation is unnecessary and fits will be noise.
- AFM3 runs can drift into FQ at high D near θ_c (paper footnote 75); flag runs whose m drops discontinuously during optimization.
- Simple update can miss phases near transitions; never report SU-only boundaries without FU/AD checks.
- Knight-move NNN terms make every evolution step and energy evaluation markedly more expensive; profile at D=3 before launching scans.
- For AD: wrap `fp_ctmrg` in `NoFixedPointError` handling (perturb and retry), and checkpoint every optimizer step.
- `transfer_matrix_spectrum` is numpy-only; compute correlation lengths in a separate post-processing pass.
- Keep numpy backend for evolution, torch only for AD; never mix configs in one state object.

---

## 9. CLAUDE.md (paste into repo)

```
Project: spin-1 BBH (+NNN) on triangular lattice with YASTN iPEPS. Read SPEC.md first; it is authoritative.
Conventions: SPEC §2 (Site=(row,col); a1=(0,1), a2=(1,0), diag=(-1,1); NNN=(1,1),(-1,2),(2,-1)).
Work milestone by milestone (SPEC §6). Do not start M(n+1) until M(n) tests pass.
Python via uv only: `uv sync` (add `--extra ad` for torch/AD); never pip/conda into the env.
Run tests: uv run pytest -q (slow ED tests: uv run pytest -m slow). CLI: uv run bbh-run config.yaml.
yastn is pinned to a git commit in pyproject.toml; M0 findings refer to that commit.
Never hard-code physics numbers in src/; reference values live in tests/ with a citation comment.
When YASTN behavior is uncertain, write a minimal test and record the result in docs/yastn_findings.md.
```
