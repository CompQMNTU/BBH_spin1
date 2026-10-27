"""Production layer: io, extrapolate, Chain stall/time exits, runs (tasks), cli (bbh-run)."""

import json
import os

import numpy as np
import pytest

from bbh_tri import cli, extrapolate, io, lattice, operators, runs, scan, states
from bbh_tri.hamiltonian import Model


# ---- extrapolate --------------------------------------------------------------------------

def test_extrapolate_paper_method():
    # midpoint of the lowest energy and the 1/D-fit intercept; error half their difference
    r = extrapolate.extrapolate([4, 5, 6], [1.6508198, 1.6438495, 1.6420273])
    D = np.array([4.0, 5, 6])
    slope, icpt = np.polyfit(1 / D, [1.6508198, 1.6438495, 1.6420273], 1)
    assert r["linear"] == pytest.approx(icpt)
    assert r["lowest"] == 1.6420273
    assert r["estimate"] == pytest.approx((icpt + 1.6420273) / 2)
    assert r["error"] == pytest.approx(abs(1.6420273 - icpt) / 2)
    assert r["estimate"] == pytest.approx(1.63268, abs=1e-5) and r["error"] == pytest.approx(0.00935, abs=1e-5)
    two = extrapolate.extrapolate([5, 4], [1.0, 2.0])  # exact line in 1/D: E = 20/D - 3
    assert two["linear"] == pytest.approx(-3.0) and max(map(abs, two["residuals"])) < 1e-12


def test_extrapolate_needs_two_bond_dimensions():
    with pytest.raises(ValueError):
        extrapolate.extrapolate([4, 4], [1.0, 1.1])
    assert set(extrapolate.subsets([6, 4, 5], [1.0, 2.0, 1.5])) == {(4, 5), (5, 6), (4, 5, 6)}
    # several seeds at one D: the subsets with a single distinct D are skipped, not fitted
    assert set(extrapolate.subsets([4, 4, 5], [2.0, 2.1, 1.5])) == {(4, 5), (4, 4, 5)}


# ---- io -----------------------------------------------------------------------------------

def test_metadata_records_versions_and_revision():
    m = io.metadata()
    assert m["yastn"] and m["numpy"] and m["python"] and m["time"]
    assert m["git"] is None or len(m["git"]) == 40


def test_state_round_trip_keeps_geometry(tmp_path):
    ops = operators.spin1("dense")
    psi = states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors("SU3"), ops)
    io.save_state(str(tmp_path / "s.pkl"), psi)
    back = io.load_state(str(tmp_path / "s.pkl"), ops.config)
    assert type(back.geometry) is type(psi.geometry) and back.sites() == psi.sites()
    for s in psi.sites():
        assert np.allclose(back[s].to_numpy(), psi[s].to_numpy())


def test_write_json_is_atomic(tmp_path):
    p = str(tmp_path / "sub" / "r.json")
    io.write_json(p, {"a": 1})
    assert io.read_json(p) == {"a": 1} and not os.path.exists(p + ".tmp")


# ---- config / cli -------------------------------------------------------------------------

def test_load_config_merges_runs_formats_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("task: variational\nsym: U1xU1\nchi: [3D2, 4D2]\nout: res/D{D}_{sym}\n"
                 "runs:\n  - {D: 4, seed: bp:SU3}\n  - {D: 5, seed: a.pkl}\n")
    e = cli.load_config(str(p), ["max_steps=7", "sym=dense"])
    assert [x["out"] for x in e] == ["res/D4_dense", "res/D5_dense"]
    assert e[1]["seed"] == "a.pkl" and e[0]["max_steps"] == 7
    e = cli.load_config(str(p), ["seed=bp:x.pkl", "D=6"])  # --set also beats the per-run entries
    assert [(x["seed"], x["D"], x["out"]) for x in e] == [("bp:x.pkl", 6, "res/D6_U1xU1")] * 2
    assert runs.chi_list(e[0]["chi"], 5) == [75, 100] and runs.chi_list(None, 4) == [48] and runs.chi_list(60, 4) == [60]


def test_templates_resolve_in_any_key_order(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("task: validate\nout: '{state}.v.json'\nruns:\n  - {state: 'r/t{theta_over_pi}.pkl', theta_over_pi: 0.25}\n")
    assert cli.load_config(str(p))[0]["out"] == "r/t0.25.pkl.v.json"
    p.write_text("task: validate\nout: '{nope}.json'\n")
    with pytest.raises(ValueError, match="unknown option"):
        cli.load_config(str(p))
    p.write_text("task: validate\na: '{b}'\nb: '{a}'\n")
    with pytest.raises(ValueError, match="cyclic"):
        cli.load_config(str(p))


def test_scientific_notation_reads_as_float(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("task: variational\ntol_grad: 1e-6\nstall_gain: 2E-8\nchi: 3D2\n")
    e = cli.load_config(str(p), ["noise=1e-4"])[0]
    assert (e["tol_grad"], e["stall_gain"], e["noise"], e["chi"]) == (1e-6, 2e-8, 1e-4, "3D2")


def test_product_seed_on_a_nondefault_cell():
    psi = runs.seed_state("SU3", 1, 0.25 * np.pi, cell="patch", cell_dims=[3, 3])
    assert len(psi.sites()) == 9


def test_nnn_options_reach_the_seed_and_its_record(tmp_path):
    """J2, K2 are task options; a bp seed with them evolves with NNN gates, which need a patch cell."""
    r = runs.run_seed({"seed": "FQ", "D": 1, "theta_over_pi": 0.3, "J2": 0.2, "K2": -0.1, "cell": "patch",
                       "cell_dims": [2, 2], "out": str(tmp_path / "fq.pkl")})
    assert (r["J2"], r["K2"]) == (0.2, -0.1)
    assert r["energy_chi_D2"] == pytest.approx(states.product_energy(Model(0.3 * np.pi, J2=0.2, K2=-0.1),
                                                                     states.seed_vectors("FQ")), abs=1e-12)
    with pytest.raises(ValueError, match="patch cell"):
        runs.seed_state("bp:FQ", 2, 0.3 * np.pi, schedule=[0.1], J2=0.2)  # sqrt3 (patch-cell runs: test_nnn.py)


def test_scan_records_without_nnn_fields_read_as_nn_sqrt3():
    old = {"n_theta": 80, "D": 4, "env": "BP"}
    assert scan.run_params(old) == {**scan.run_params({**old, "J2": 0.0, "K2": 0.0, "cell": "sqrt3"}), "cell_dims": None}
    assert scan.run_params({**old, "J2": 0.1})["J2"] == 0.1


def test_options_reject_unknown_and_missing_keys():
    with pytest.raises(ValueError, match="unknown option"):
        runs.options({"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": "x", "max_step": 3}, runs.VARIATIONAL, "variational")
    with pytest.raises(ValueError, match="missing"):
        runs.options({"D": 2}, runs.VARIATIONAL, "variational")


def test_cli_dry_run_and_index(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("task: extrapolate\nruns:\n  - {points: [[4, 2.0], [5, 1.0]]}\n  - {points: [[2, 1.0], [3, 0.5]]}\n")
    assert cli.main([str(p), "--index", "1", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["points"] == [[2, 1.0], [3, 0.5]]
    assert cli.main([str(p), "--index", "0"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["all"]["linear"] == pytest.approx(-3.0)
    with pytest.raises(SystemExit):
        cli.main([str(p), "--index", "5"])


def test_extrapolate_task_reads_result_files(tmp_path):
    for D, Es in ((4, {"32": 2.1, "64": 2.0}), (5, {"50": 1.2, "100": 1.0})):
        io.write_json(str(tmp_path / f"D{D}.json"), {"D": D, "validation": {"energies": Es}})
    r = runs.run_extrapolate({"results": [str(tmp_path / "D4.json"), str(tmp_path / "D5.json")], "reference": [-3.0, 0.1],
                              "out": str(tmp_path / "x.json")})
    assert r["used"] == [(4, 2.0), (5, 1.0)] and r["reference"]["consistent"] is True
    assert io.read_json(str(tmp_path / "x.json"))["reference"]["consistent"] is True
    r = runs.run_extrapolate({"results": [str(tmp_path / "D4.json")], "chi": 32, "points": [[6, 0.9], [5, 1.5]]})
    assert (4, 2.1) in r["used"]
    assert set(r["subsets"]) == {"4,5", "5,6", "4,5,6"}  # consecutive runs of the sorted D
    with pytest.raises(ValueError, match="no energy at chi"):
        runs.run_extrapolate({"results": [str(tmp_path / "D5.json")], "chi": 32, "points": [[6, 0.9]]})
    io.write_json(str(tmp_path / "bad.json"), {"D": 6, "energies": {"36": 1.0, "72": 0.9},
                                                "ctm_converged": {"36": True, "72": False}})
    with pytest.raises(ValueError, match="did not converge"):
        runs.run_extrapolate({"results": [str(tmp_path / "bad.json")], "points": [[4, 2.0]]})
    r = runs.run_extrapolate({"results": [str(tmp_path / "bad.json")], "chi": 36, "points": [[4, 2.0]]})
    assert (6, 1.0) in r["used"]


def test_extrapolate_task_uses_the_best_seed_per_D_and_one_ladder(tmp_path):
    r = runs.run_extrapolate({"points": [[4, 2.0], [4, 2.1], [5, 1.0]]})  # an inferior D = 4 seed is ignored
    assert r["used"] == [(4, 2.0), (5, 1.0)] and r["all"]["estimate"] == pytest.approx(-1.0)
    for name, theta, sym in (("a", 0.25, "U1xU1"), ("b", 0.25, "dense")):
        io.write_json(str(tmp_path / f"{name}.json"), {"D": 4 if name == "a" else 5, "theta_over_pi": theta, "sym": sym,
                                                        "validation": {"energies": {"32": 1.0}}})
    with pytest.raises(ValueError, match="mix"):
        runs.run_extrapolate({"results": [str(tmp_path / "a.json"), str(tmp_path / "b.json")]})
    io.write_json(str(tmp_path / "c.json"), {"D": 5, "theta_over_pi": 0.25, "sym": "U1xU1", "J2": 0.1,
                                              "validation": {"energies": {"32": 1.0}}})
    with pytest.raises(ValueError, match="mix"):  # same theta and symmetry, another J2
        runs.run_extrapolate({"results": [str(tmp_path / "a.json"), str(tmp_path / "c.json")]})
    with pytest.raises(ValueError, match="two or more D"):
        runs.run_extrapolate({"points": [[4, 2.0], [4, 2.1]]})


# ---- scan ---------------------------------------------------------------------------------

def test_best_per_theta_prefers_seeds_within_tie(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [{"theta_k": 0, "init": "random", "energy": 1.0}, {"theta_k": 0, "init": "FQ", "energy": 1.0 + 1e-9},
            {"theta_k": 1, "init": "SU3", "energy": 2.0}, {"theta_k": 1, "init": "FM", "energy": 1.5}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [(r["theta_k"], r["init"]) for r in scan.best_per_theta(str(p))] == [(0, "FQ"), (1, "FM")]


# ---- tasks that need torch ----------------------------------------------------------------

torch_needed = pytest.mark.ad


@pytest.fixture
def small_chain():
    pytest.importorskip("torch")
    from bbh_tri import evolve, variational as var
    ops = operators.spin1("dense")
    psi = states.random_product_peps(lattice.geometry("sqrt3"), ops, seed=1)
    evolve.evolve(psi, Model(np.pi / 4), ops, 2, schedule=(0.1,), max_steps=5)
    tops = var.torch_ops()
    return var.Chain(var.convert(psi, tops.config), Model(np.pi / 4), tops, chi=8)


def _fake_steps(ch, monkeypatch, gains, evals):
    def fake_step():
        k = len(ch.history)
        e = {"step": k + 1, "energy": 1.0 - sum(gains[:k + 1]), "grad_norm": 1.0, "evals": evals[k],
             "zeroed_pairs": 0, "zeroed_grad": False}
        ch.history.append(e)
        return e
    monkeypatch.setattr(ch, "step", fake_step)


@torch_needed
def test_run_stops_on_a_stalled_line_search(small_chain, monkeypatch):
    # 5 steps x 20 evaluations for 1e-8 each: stalled; tol_E is 0 so only the stall test can stop it
    _fake_steps(small_chain, monkeypatch, [1e-3] * 3 + [1e-8] * 10, [1] * 3 + [20] * 10)
    assert small_chain.run(max_steps=13, tol_E=0.0) == "stalled"
    assert len(small_chain.history) == 8


@torch_needed
def test_stall_wins_over_flat_energy_convergence(small_chain, monkeypatch):
    # flat energies (span < tol_E) at 20 evaluations per step and a large gradient: a stall, not
    # convergence, so that run_variational moves on to the next chi
    _fake_steps(small_chain, monkeypatch, [1e-12] * 10, [20] * 10)
    assert small_chain.run(max_steps=10) == "stalled"


@torch_needed
def test_stall_exit_passes_the_final_fd_gate(small_chain, monkeypatch):
    # a stall on an unchecked zero-mode gradient: the FD check fails, the step is rolled back and
    # retried (perturbed); the next stall passes the check and ends the run
    ch = small_chain
    checks, perturbs = [], []

    def fake_step():
        k = len(ch.history)
        e = {"step": k + 1, "energy": 1.0 - 1e-12 * k, "grad_norm": 1.0, "evals": 20,
             "zeroed_pairs": 1, "zeroed_grad": True}
        ch.history.append(e)
        return e
    monkeypatch.setattr(ch, "step", fake_step)
    monkeypatch.setattr(ch, "check_gradient", lambda seed: (checks.append(seed), (len(checks) > 1, [(0.0, 1.0)], 0.0))[1])
    monkeypatch.setattr(ch, "perturb", lambda seed=None: perturbs.append(seed))
    assert ch.run(max_steps=20) == "stalled"
    assert len(checks) == 2 and len(perturbs) == 1 and "fd_check" in ch.history[-1]


@torch_needed
def test_cheap_slow_tail_is_not_a_stall(small_chain, monkeypatch):
    # one evaluation per step and 1e-8 per step: slow but healthy
    _fake_steps(small_chain, monkeypatch, [1e-8] * 12, [1] * 12)
    assert small_chain.run(max_steps=12, tol_E=0.0) == "max_steps"


@torch_needed
def test_run_honours_the_deadline(small_chain, monkeypatch):
    _fake_steps(small_chain, monkeypatch, [1e-3] * 5, [1] * 5)
    assert small_chain.run(max_steps=5, deadline=0.0) == "time" and small_chain.history == []


@torch_needed
def test_checkpoint_loads_as_a_state(small_chain, tmp_path):
    ck = str(tmp_path / "c.pt")
    small_chain.save(ck)
    a = io.load_state(ck, operators.spin1("dense").config)
    b = small_chain.numpy_peps()
    for s in b.sites():
        assert np.allclose(a[s].to_numpy(), b[s].to_numpy())


@torch_needed
@pytest.mark.slow
def test_variational_task_escalates_chi_and_resumes(tmp_path):
    pytest.importorskip("torch")
    out = str(tmp_path / "new_dir" / "r")  # the output directory is created
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": out, "chi": [8, 12], "max_steps": 3,
           "stall_window": 1, "stall_min_evals": 1, "stall_gain": 1.0,  # every step counts as a stall
           "validate_chi": [8]}
    rec = runs.run_variational(cfg, log=lambda s: None)
    # the chi = 12 stage may end either way (D = 2 at pi/4 sits at the BP minimum: 'grad' or 'stalled')
    assert [s["chi"] for s in rec["stages"]] == [8, 12] and rec["stages"][0]["status"] == "stalled"
    assert os.path.exists(out + ".pt") and os.path.exists(out + ".chi12.pt") and os.path.exists(out + ".pkl")
    assert {h["chi"] for h in rec["history"]} == {8, 12} and rec["metadata"]["yastn"]
    assert rec["energy_opt"] == [h for h in rec["history"] if h["chi"] == 12][-1]["energy"]
    assert set(rec["validation"]["energies"]) == {8}
    again = runs.run_variational(cfg, log=lambda s: None)  # finished: rewrites the outputs only
    assert again["history"] == rec["history"]
    assert len(again["sessions"]) == 2 and again["init"]["seed"] == "bp"  # every session is kept
    assert again["sessions"][0]["options"]["stall_gain"] == 1.0 and {s["session"] for s in again["stages"]} == {0}
    seed = runs.run_seed({"seed": f"bp:{out}.pkl", "D": 2, "theta_over_pi": 0.25, "schedule": [0.1],
                          "out": str(tmp_path / "s.pkl")})  # a variational output's record is found
    assert seed["seed_provenance"]["task"] == "variational" and "history" not in seed["seed_provenance"]
    with pytest.raises(ValueError, match="was started with"):  # another seed would relabel the old chain
        runs.run_variational({**cfg, "rng": 1}, log=lambda s: None)
    os.remove(out + ".run.json")  # a chain from before OUT.run.json: the first run's options bind
    assert runs.run_variational(cfg, log=lambda s: None)["init"]["unverified"]
    with pytest.raises(ValueError, match="was started with"):
        runs.run_variational({**cfg, "rng": 1}, log=lambda s: None)
    with pytest.raises(ValueError, match="was made for"):  # finished stages are identity-checked too
        runs.run_variational({**cfg, "theta_over_pi": 0.3}, log=lambda s: None)
    with pytest.raises(ValueError, match="was made for"):  # the NNN couplings belong to the model
        runs.run_variational({**cfg, "J2": 0.1}, log=lambda s: None)


@torch_needed
@pytest.mark.slow
def test_variational_task_resumes_a_step_limit(tmp_path):
    pytest.importorskip("torch")
    out = str(tmp_path / "r")
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": out, "chi": 8, "max_steps": 1, "validate": False}
    assert runs.run_variational(cfg, log=lambda s: None)["status"] == "max_steps"
    rec = runs.run_variational({**cfg, "max_steps": 2}, log=lambda s: None)
    assert len(rec["history"]) == 2 and [s["steps"] for s in rec["stages"]] == [2]


@torch_needed
def test_variational_task_refuses_another_ad_on_resume(tmp_path):
    pytest.importorskip("torch")
    out = str(tmp_path / "r")
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "FM", "out": out, "chi": 8, "max_steps": 1, "max_hours": 1e-9}
    with pytest.raises(ValueError, match="ad must be one of"):
        runs.run_variational({**cfg, "ad": "implicit"}, log=lambda s: None)
    runs.run_variational(cfg, log=lambda s: None)  # a checkpoint before the first step, ad = 'unrolled'
    with pytest.raises(ValueError, match=r"made with ad = 'unrolled', not 'fp'.*start another out with seed"):
        runs.run_variational({**cfg, "ad": "fp"}, log=lambda s: None)


@torch_needed
@pytest.mark.slow
def test_variational_task_runs_an_fp_chain(tmp_path):
    pytest.importorskip("torch")
    import torch
    out = str(tmp_path / "r")
    seed = str(tmp_path / "unrolled")
    runs.run_variational({"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": seed, "chi": 8, "max_steps": 1,
                          "validate": False}, log=lambda s: None)
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": seed + ".pt", "out": out, "chi": 8, "max_steps": 2, "ad": "fp",
           "validate": False}  # an unrolled chain's state continued with the fp gradient
    rec = runs.run_variational(cfg, log=lambda s: None)
    assert len(rec["history"]) == 2 and rec["options"]["ad"] == "fp"
    assert torch.load(out + ".pt", weights_only=False)["meta"]["ad"] == "fp"
    assert rec["history"][-1]["energy"] <= rec["history"][0]["energy"]


@torch_needed
def test_variational_task_stops_cleanly_before_the_first_step(tmp_path):
    pytest.importorskip("torch")
    out = str(tmp_path / "r")
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "FM", "out": out, "chi": 8, "max_steps": 1, "max_hours": 1e-9}
    rec = runs.run_variational(cfg, log=lambda s: None)
    assert rec["status"] == "time" and rec["history"] == [] and "validation" not in rec
    assert os.path.exists(out + ".pt") and os.path.exists(out + ".pkl")
    rec = runs.run_variational({**cfg, "max_hours": None, "validate": False}, log=lambda s: None)
    assert rec["status"] == "max_steps" and len(rec["history"]) == 1


@torch_needed
def test_seed_evolution_stops_at_the_deadline_and_resumes(tmp_path, capsys):
    pytest.importorskip("torch")
    out = str(tmp_path / "r")
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": out, "chi": 8, "max_steps": 1, "max_hours": 1e-9,
           "schedule": [0.1, 0.05], "validate": False}
    rec = runs.run_variational(cfg, log=lambda s: None)  # cut after the first Trotter step
    assert rec["status"] == "time" and rec["seed_partial"] == out + ".seed.pkl" and not os.path.exists(out + ".pt")
    assert io.read_json(out + ".seed.pkl.json")["taus_done"] == 0
    rec = runs.run_variational({**cfg, "max_hours": None}, log=lambda s: None)
    assert "continuing" in capsys.readouterr().out  # the cut tau is rerun from the kept state
    assert rec["status"] == "max_steps" and len(rec["sessions"]) == 2 and not os.path.exists(out + ".seed.pkl")
    runs.run_variational({**cfg, "out": out + "2"}, log=lambda s: None)  # other seed options discard the kept evolution
    capsys.readouterr()
    runs.run_variational({**cfg, "out": out + "2", "rng": 1, "max_hours": None}, log=lambda s: None)
    assert "continuing" not in capsys.readouterr().out and len(io.read_json(out + "2.run.json")["sessions"]) == 1


@torch_needed
@pytest.mark.slow
def test_time_exit_before_a_later_stage_reports_no_energy_for_it(tmp_path):
    pytest.importorskip("torch")
    out = str(tmp_path / "r")
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": out, "chi": [8], "max_steps": 2, "validate": False,
           "stall_window": 1, "stall_min_evals": 1, "stall_gain": 1.0}
    assert runs.run_variational(cfg, log=lambda s: None)["status"] == "stalled"
    rec = runs.run_variational({**cfg, "chi": [8, 12], "max_hours": 1e-9}, log=lambda s: None)  # escalate, out of time
    assert rec["status"] == "time" and rec["chi_opt"] == 12 and rec["energy_opt"] is None
    assert os.path.exists(out + ".chi12.pt")


@torch_needed
def test_variational_task_skips_validation_after_the_deadline(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import time as real_time

    class Clock:  # runs' clock: the run starts now, and by validation two hours have passed
        calls = 0

        @classmethod
        def time(cls):
            cls.calls += 1
            return real_time.time() + (0 if cls.calls <= 2 else 7200)
    monkeypatch.setattr(runs, "time", Clock)
    out = str(tmp_path / "r")
    rec = runs.run_variational({"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": out, "chi": 8, "max_steps": 1,
                                "max_hours": 1.0}, log=lambda s: None)
    assert rec["status"] == "max_steps" and "validation" not in rec and "validation_skipped" in rec
    assert rec["cell"] == {"type": "TriangularLattice", "dims": [3, 3], "unique_sites": 3, "boundary": "infinite"}


@torch_needed
def test_validation_stops_at_the_deadline_and_reserve_shortens_the_optimization(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from bbh_tri import variational as var
    import time as real_time
    t0 = real_time.time()  # Chain.run checks the deadline against the real clock
    now = [t0]
    monkeypatch.setattr(runs, "time", type("Clock", (), {"time": staticmethod(lambda: now[0])}))
    real_validate, seen = var.validate, []

    def validate(psi, model, chis, **kw):  # each chi takes two hours
        seen.append(chis)
        now[0] += 7200
        return real_validate(psi, model, chis, **kw)
    monkeypatch.setattr(var, "validate", validate)
    deadlines = []
    real_run = var.Chain.run
    monkeypatch.setattr(var.Chain, "run", lambda self, **kw: deadlines.append(kw["deadline"]) or real_run(self, **kw))
    cfg = {"theta_over_pi": 0.25, "D": 2, "seed": "bp", "out": str(tmp_path / "r"), "chi": 8, "max_steps": 1,
           "max_hours": 1.5, "validate_chi": [6, 8], "validate_reserve_hours": 0.5}
    rec = runs.run_variational(cfg, log=lambda s: None)
    assert deadlines == [t0 + 3600]  # 1.5 h minus the 0.5 h reserve
    assert seen == [[6]] and "validation" not in rec  # the 8 would start after the deadline
    assert set(rec["validation_partial"]["energies"]) == {6} and "validation_skipped" in rec
    # cut at the optimization cutoff: the reserve validates the current state, apart from "validation"
    seen.clear()
    now[0] = t0 = real_time.time()
    rec = runs.run_variational({**cfg, "out": str(tmp_path / "q"), "max_hours": 1.0, "validate_reserve_hours": 1.0,
                                "validate_chi": [6], "seed": "FM"}, log=lambda s: None)  # FM: no seed evolution to cut
    assert rec["status"] == "time" and seen == [[6]]
    assert "validation" not in rec and set(rec["validation_interim"]["energies"]) == {6}


def test_validate_task_records_D(tmp_path):
    p = str(tmp_path / "d1.pkl")
    io.save_state(p, states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors("FM"), operators.spin1("dense")))
    r = runs.run_validate({"state": p, "theta_over_pi": 0.75, "chi": [1, 2], "out": str(tmp_path / "v.json"),
                           "max_sweeps": 300})
    saved = io.read_json(str(tmp_path / "v.json"))["options"]
    assert saved["max_sweeps"] == 300 and saved["corner_tol"] == runs.VALIDATE["corner_tol"]
    assert r["D"] == 1 and r["energies"][2] == pytest.approx(3 * (np.cos(0.75 * np.pi) + np.sin(0.75 * np.pi)))
    assert runs._result_point(str(tmp_path / "v.json"), "max") == (1, pytest.approx(r["energies"][2]),
                                                                    (0.75, "dense", 0.0, 0.0))
    r2 = runs.run_validate({"state": p, "theta_over_pi": 0.75, "J2": -0.2, "K2": 0.1, "chi": [1]})
    assert r2["J2"] == -0.2 and r2["energies"][1] == pytest.approx(r["energies"][2] + 3 * (-0.2 + 0.1))  # FM: S.S = 1


def test_seed_task_truncates_a_saved_state(tmp_path):
    p1 = str(tmp_path / "d1.pkl")
    io.save_state(p1, states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors("SU3"),
                                          operators.spin1("U1xU1")))
    r = runs.run_seed({"seed": f"bp:{p1}", "D": 2, "theta_over_pi": 0.25, "sym": "U1xU1", "schedule": [0.1],
                       "out": str(tmp_path / "d2.pkl")})
    assert r["bond_dims"] == [2]
    r = runs.run_seed({"seed": f"bp:{r['out']}", "D": 1, "theta_over_pi": 0.25, "sym": "U1xU1", "schedule": [0.1],
                       "out": str(tmp_path / "d1b.pkl")})
    assert r["bond_dims"] == [1]
    side = io.read_json(r["out"] + ".json")  # each seed writes how it was made, and links its source's record
    assert side["options"]["D"] == 1 and side["metadata"]["yastn"]
    assert side["seed_provenance"]["options"]["D"] == 2 and side["seed_provenance"]["seed_provenance"] is None


@torch_needed
def test_seed_and_line_scan_tasks(tmp_path):
    pytest.importorskip("torch")
    p1 = str(tmp_path / "d1.pkl")
    io.save_state(p1, states.product_peps(lattice.geometry("sqrt3"), states.seed_vectors("SU3"),
                                          operators.spin1("U1xU1")))
    r = runs.run_seed({"seed": f"bp:{p1}", "D": 2, "theta_over_pi": 0.25, "sym": "U1xU1", "schedule": [0.1],
                       "out": str(tmp_path / "d2.pkl")})
    assert r["bond_dims"] == [2]
    ls = runs.run_line_scan({"state": r["out"], "theta_over_pi": 0.25, "sym": "U1xU1", "chi": 8, "steps": [1e-4, 1e-3]})
    assert [x["t"] for x in ls["scan"]] == [1e-4, 1e-3] and ls["grad_norm"] > 0 and ls["theta_over_pi"] == 0.25
    assert ls["scan"][0]["dE"] < 0  # a small step along -g lowers E at an unconverged state
