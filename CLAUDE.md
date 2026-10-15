# CLAUDE.md

Spin-1 bilinear-biquadratic model (+ NNN J₂/K₂) on the triangular lattice, studied with YASTN iPEPS.
Reference: Niesen & Corboz, PRB 97, 245146 (2018). **SPEC.md is authoritative**; read it before changing code.

## Commands

Python via **uv only**. Never pip/conda into the env.

```
uv sync                                # base env (numpy backend)
uv sync --extra ad                     # + torch (CPU wheel index) for AD / fp_ctmrg
uv run pytest -q                       # default suite, skips `slow` (~15 s)
uv run pytest -m "slow or not slow"    # everything incl. FD gradient checks (~40 s)
uv run pytest -m slow                  # slow only (ED, gradient checks)
uv run pytest tests/test_routing.py -k "diag and BP"   # single test selection
```

Markers: `slow` (deselected by default via `addopts`), `ad` (needs `--extra ad`).
`tests/conftest.py` pins OMP/OPENBLAS/MKL to 1 thread; do the same in any script or SLURM job.
Planned CLI (not implemented yet): `uv run bbh-run config.yaml`.

## Layout

- `SPEC.md`: goals, conventions (§2), reference numbers (§3), algorithms (§4), target package layout (§5), milestones (§6), pitfalls (§8).
- `docs/yastn_findings.md`: verified YASTN behavior and bugs at the pinned commit. `docs/ad_workflow.md`: design of the AD/fp_ctmrg ground-state workflow.
- `src/bbh_tri/`: the package. So far only `yastn_compat.py` (workarounds for YASTN bugs). Modules listed in SPEC §5 are added milestone by milestone.
- `tests/`: `test_routing.py` (U1), `test_measure.py` (U2), `test_fp_ctmrg.py` (U4), `test_yastn_canaries.py` (pins current upstream behavior, including bugs), `test_yastn_compat.py` (our workarounds), `yastn_utils.py` (test-only helpers).

## Working rules

- Milestones (SPEC §6): M0 done; M4's AD blocker fixed (PR #4). Do not start M(n+1) until M(n) tests pass.
- Conventions (SPEC §2): `Site = (row, col)`; a₁ = (0,1), a₂ = (1,0), NN diagonal = (−1,1); NNN = (1,1), (−1,2), (2,−1). Energy per site, form (1) of the paper. Dense Spin1 basis order is (+1, 0, −1).
- Never hard-code physics numbers in `src/`. Reference values live in `tests/` with a citation comment.
- When YASTN behavior is uncertain, write a minimal test, record the result in `docs/yastn_findings.md`, and add a canary to `tests/test_yastn_canaries.py`. If a canary fails after a yastn bump, update the findings and SPEC §1 before the test.
- yastn is pinned to a git commit in `pyproject.toml`; all findings refer to that commit.
- Numpy backend for evolution, torch only for AD; never mix configs in one state.

## YASTN pitfalls (details: docs/yastn_findings.md)

- Never pass `TriangularLattice` to `gates.distribute` or call all-bond defaults: `measure_nn()`, `truncate_(bond=None)`, `get_bond_dimensions()`, and `bonds()` without a direction (raises on `full_patch`). Iterate `bonds('h')`, `bonds('v')`, `bonds('d')` explicitly.
- Diagonal and NNN gates must be routed along explicit square-NN paths (`gate._replace(sites=path)`); every bond on the path is truncated. `distribute(symmetrize=True)` does not mirror custom gates: use τ/2 and append the reversed list yourself.
- Measure the diagonal bond with `measure_2x2`, `measure_nsite_exact` or `rdm2x2_anti_diagonal`. `measure_rdm_nn/diag/2x2` crash on dense configs; Finite OBC patches need `TriangularLattice(dims=..., boundary='obc', full_patch=True)` (without `full_patch`, `dims` is ignored), and their `bonds('d')` contains `None`.
- **AD:** run every forward and backward pass inside `bbh_tri.yastn_compat.exact_svd_backward()`. yastn's SVD backward regularizes with an absolute 1e-12 and gives wrong gradients otherwise. Treat `DegenerateSVDWarning` as an error: degenerate singular values make the gradient unreliable. Always truncate with `opts_svd['tol'] > 0` (kept exact zeros are undetectable). `fp_ctmrg` returns only `env`.
- **Saving:** use `bbh_tri.yastn_compat.peps_to_dict(psi)`. `psi.to_dict()` drops `dims/boundary/full_patch`, so non-default cells reload with the wrong geometry.
- NTU is slow (D=3 NTU didn't finish 100 steps in 8 min); EnvBP is the practical default at small D. Seed with `ops.random_seed(seed=...)`.

## Git / GitHub

- Commit, push, open PRs and merge only when asked. Branch off `main`.
- **A PR may be merged only after a Codex review has cleared.** Run it on the final PR head, then verify each finding against the code. Fix real ones on the same branch and re-review; record dismissed ones with a reason in a PR comment. Merge only when no unresolved findings remain, even if the user asks to merge earlier: report the outstanding findings instead.
- Running Codex here: its bubblewrap sandbox cannot start on this machine, so `codex review --base main` cannot read the repo. Pass the diff inline instead (omit `uv.lock`):
  ```
  { echo "Review this PR of CompQMNTU/BBH_spin1. Your shell sandbox is unavailable; CLAUDE.md and the diff are inline. Report concrete defects (physics/maths errors, wrong claims, bugs, test gaps) with file:line, most severe first."
    echo "=== CLAUDE.md ==="; cat CLAUDE.md
    echo "=== git diff main...HEAD ==="; git diff main...HEAD -- . ':!uv.lock'; } | codex review - > codex_review.log 2>&1
  ```
  Takes several minutes; run it in the background. Output ending in "Review blocked" or "could not be completed" does not count as cleared.
- `gh pr edit` fails (Projects-classic GraphQL deprecation). Edit PR bodies with `gh api -X PATCH repos/CompQMNTU/BBH_spin1/pulls/N -F body=@file`.
