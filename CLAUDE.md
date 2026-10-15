Project: spin-1 BBH (+NNN) on triangular lattice with YASTN iPEPS. Read SPEC.md first; it is authoritative.
Conventions: SPEC §2 (Site=(row,col); a1=(0,1), a2=(1,0), diag=(-1,1); NNN=(1,1),(-1,2),(2,-1)).
Work milestone by milestone (SPEC §6). Do not start M(n+1) until M(n) tests pass.
Python via uv only: `uv sync` (add `--extra ad` for torch/AD); never pip/conda into the env.
Run tests: uv run pytest -q (slow ED tests: uv run pytest -m slow). CLI: uv run bbh-run config.yaml.
yastn is pinned to a git commit in pyproject.toml; M0 findings refer to that commit.
Never hard-code physics numbers in src/; reference values live in tests/ with a citation comment.
When YASTN behavior is uncertain, write a minimal test and record the result in docs/yastn_findings.md.
YASTN pitfalls (docs/yastn_findings.md): never pass TriangularLattice to gates.distribute or call all-bond defaults (measure_nn(), truncate_(bond=None)); route diagonal/NNN gates along explicit NN paths; measure the diagonal with measure_2x2 / measure_nsite_exact.
