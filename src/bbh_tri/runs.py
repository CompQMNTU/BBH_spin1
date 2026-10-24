"""Production tasks behind ``bbh-run`` (cli.py) and the scripts in ``scripts/``.

Every task takes one flat dict of options (a YAML config after ``cli.load_config``), checks it
against its defaults (an unknown key raises: typos must not silently fall back to a default) and
returns a result dict. Tasks:

- ``variational``: AD chain(s) for one (theta, D, sym, seed); resumable; optional chi escalation
  after a stall; validation; writes OUT.pt (checkpoint), OUT.pkl (state), OUT.json (record).
- ``validate``: re-measure a saved state (pickle or checkpoint) at several chi.
- ``seed``: build a seed state by EnvBP evolution at D (e.g. a D state from a D+1 or D-1 state).
- ``line_scan``: energy along -g at a state (stall diagnosis, docs/ad_workflow.md §2).
- ``extrapolate``: D -> inf extrapolation (paper method) from points or result files.
- ``scan``: theta scan by imaginary-time evolution (scan.py).

Seed specs (``seed``): a path (.pkl state or .pt checkpoint; dense states are padded up to D,
symmetric ones must already be at D), ``bp`` (random product state evolved at D, dense),
``bp:NAME`` (product seed NAME = FM, FQ, AFQ, SU3 evolved at D), ``bp:PATH`` (saved state evolved
at D: grows or truncates its bonds), ``random`` or NAME (product state, padded; dense only).
"""

import json
import os
import time

import numpy as np

from . import io

# ---- options ------------------------------------------------------------------------------

COMMON = {"task": None, "threads": 1}

VARIATIONAL = {**COMMON, "theta_over_pi": None, "D": None, "seed": None, "out": None, "sym": "dense",
               "cell": "sqrt3", "cell_dims": None, "chi": None, "max_steps": 200, "noise": 1e-3, "rng": 0, "max_sweeps": 100,
               "device": "cpu", "max_hours": None, "tol_E": 1e-8, "tol_grad": 1e-6,
               "stall_window": 5, "stall_min_evals": 25, "stall_gain": 2e-8,
               "validate": True, "validate_chi": None, "validate_reserve_hours": 0, "schedule": None}

# options that make the initial state; a resumed chain must have been started with the same ones
INIT_KEYS = ("seed", "noise", "rng", "schedule", "cell", "cell_dims")

VALIDATE = {**COMMON, "state": None, "theta_over_pi": None, "sym": "dense", "chi": None, "out": None,
            "corner_tol": 1e-10, "max_sweeps": 500}

SEED = {**COMMON, "seed": None, "D": None, "theta_over_pi": None, "sym": "dense", "cell": "sqrt3", "cell_dims": None,
        "out": None,
        "noise": 1e-3, "rng": 0, "schedule": None}

LINE_SCAN = {**COMMON, "state": None, "theta_over_pi": None, "sym": "dense", "chi": None, "out": None,
             "steps": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "max_sweeps": 60, "device": "cpu"}

EXTRAPOLATE = {**COMMON, "points": None, "results": None, "chi": "max", "reference": None, "out": None}

SCAN = {**COMMON, "D": None, "n_theta": 80, "env": "BP", "out": None, "workers": 8, "schedule": None,
        "only_best": None, "from_dir": None, "save_dir": None}

REQUIRED = {"variational": ("theta_over_pi", "D", "seed", "out"), "validate": ("state", "theta_over_pi", "chi"),
            "seed": ("seed", "D", "theta_over_pi", "out"), "line_scan": ("state", "theta_over_pi", "chi"),
            "extrapolate": (), "scan": ("D", "out")}


def options(cfg, defaults, task):
    """``defaults`` updated with ``cfg``; raises on unknown or missing keys."""
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"task {task!r}: unknown option(s) {sorted(unknown)}; accepted: {sorted(defaults)}")
    o = {**defaults, **cfg}
    missing = [k for k in REQUIRED[task] if o.get(k) is None]
    if missing:
        raise ValueError(f"task {task!r}: missing option(s) {missing}")
    return o


def chi_list(chi, D):
    """Optimization chi stages: None -> [3 D^2]; an int or a list of ints; 'kD2' means k D^2."""
    if chi is None:
        return [3 * D * D]
    items = chi if isinstance(chi, (list, tuple)) else [chi]
    out = []
    for c in items:
        if isinstance(c, str) and c.upper().endswith("D2"):
            c = int(float(c[:-2]) * D * D)
        out.append(int(c))
    return out


# ---- seeds --------------------------------------------------------------------------------

def seed_state(spec, D, theta, sym="dense", noise=1e-3, rng=0, cell="sqrt3", schedule=None, cell_dims=None):
    """Numpy-backend PEPS at bond dimension D from a seed spec (module docstring). ``cell``,
    ``cell_dims``: ``lattice.geometry`` arguments for product-state seeds (a saved state brings its
    own geometry)."""
    from . import evolve, lattice, operators, states, variational as var
    from .hamiltonian import Model

    ops = operators.spin1(sym)
    geo = None if os.path.exists(spec) or (spec.startswith("bp:") and os.path.exists(spec[3:])) \
        else lattice.geometry(cell, dims=cell_dims)
    kw = {"schedule": tuple(schedule)} if schedule else {}
    if os.path.exists(spec):
        psi = io.load_state(spec, ops.config)
    elif spec == "bp":
        psi = states.random_product_peps(geo, ops, seed=rng)
        evolve.evolve(psi, Model(theta), ops, D, **kw)
    elif spec.startswith("bp:"):
        src = spec[3:]
        psi = io.load_state(src, ops.config) if os.path.exists(src) else states.product_peps(geo, states.seed_vectors(src), ops)
        evolve.evolve(psi, Model(theta), ops, D, **kw)
    elif spec == "random":
        psi = states.random_product_peps(geo, ops, seed=rng)
    else:
        psi = states.product_peps(geo, states.seed_vectors(spec), ops)
    dims = [d for s in psi.sites() for d in psi[s].get_shape()[:4]]
    if max(dims) > D:
        raise ValueError(f"seed has bond dimension {max(dims)} > D = {D} (use bp:PATH to truncate)")
    if min(dims) < D and sym == "dense":
        psi = var.pad(psi, D, noise=noise, seed=rng)
    elif max(dims) < D:  # the checkpoint would record a smaller D and refuse to resume with this D
        raise ValueError(f"symmetric seed reaches only bond dimension {max(dims)} < D = {D} "
                         "(e.g. an eigenstate seed that EnvBP does not entangle); use another seed or D")
    elif min(dims) < D:
        print(f"warning: symmetric seed has bond dimensions down to {min(dims)} < D = {D}; it stays so", flush=True)
    return psi


def run_seed(cfg):
    o = options(cfg, SEED, "seed")
    from . import measure, operators, evolve
    from .hamiltonian import Model
    theta = np.pi * o["theta_over_pi"]
    psi = seed_state(o["seed"], o["D"], theta, o["sym"], o["noise"], o["rng"], o["cell"], o["schedule"], o["cell_dims"])
    ops = operators.spin1(o["sym"])
    E, _ = measure.energy(psi, measure.ctm_env(psi, evolve.default_chi(o["D"])), ops, Model(theta))
    io.save_state(o["out"], psi)
    dims = sorted({d for s in psi.sites() for d in psi[s].get_shape()[:4]})
    rec = {"task": "seed", "out": o["out"], "energy_chi_D2": E, "bond_dims": dims, "cell": _cell(psi),
           "seed_provenance": _seed_provenance(o["seed"]), "options": o, "metadata": io.metadata()}
    io.write_json(f"{o['out']}.json", rec)  # sidecar: how the state was made
    return rec


def _seed_provenance(spec):
    """The record of the saved state a seed spec names, if any, so that a chain of seeds can be
    followed back to its origin: PATH.json (seed task sidecar) or, for the outputs of a variational
    run OUT (OUT.pkl, OUT.pt, OUT.chiN.pt), OUT.json without its history."""
    import re
    path = spec[3:] if isinstance(spec, str) and spec.startswith("bp:") else spec
    if not isinstance(path, str):
        return None
    for side in (f"{path}.json", re.sub(r"(\.chi\d+)?\.(pkl|pt)$", "", path) + ".json"):
        if side != path and os.path.exists(side):
            return {k: v for k, v in io.read_json(side).items() if k != "history"}
    return None


# ---- variational --------------------------------------------------------------------------

def _stage_checkpoint(out, chis, k):
    """Stage 0 keeps the historical name OUT.pt; later stages OUT.chiN.pt."""
    return f"{out}.pt" if k == 0 else f"{out}.chi{chis[k]}.pt"


def run_variational(cfg, log=print):
    """One AD optimization; resumable. With several chi (``chi: [3D2, 4D2]``) a stalled stage is
    continued at the next chi from its last state (docs/ad_workflow.md §2). Returns the record
    written to OUT.json (``status`` of the last stage; 'time' = stopped by ``max_hours``, resume
    by running the same config again)."""
    o = options(cfg, VARIATIONAL, "variational")
    import torch
    import yastn.tn.fpeps as fpeps

    from . import operators, variational as var
    from .hamiltonian import Model

    torch.set_num_threads(int(o["threads"]))
    D, out, theta = int(o["D"]), o["out"], np.pi * o["theta_over_pi"]
    chis = chi_list(o["chi"], D)
    model = Model(theta)
    ops = var.torch_ops(o["sym"], device=o["device"])
    np_config = operators.spin1(o["sym"]).config
    deadline = time.time() + 3600 * float(o["max_hours"]) if o["max_hours"] else None
    # the optimization stops early enough to leave validate_reserve_hours for the validation
    opt_deadline = deadline - 3600 * float(o["validate_reserve_hours"] or 0) if deadline is not None else None
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)  # Chain.save does not create it
    run_info = _run_provenance(out, o)  # checked here, written once the checkpoints pass too
    stages_path = f"{out}.stages.json"
    stages = io.read_json(stages_path) if os.path.exists(stages_path) else []
    if [s["chi"] for s in stages] != chis[:len(stages)]:
        raise ValueError(f"{stages_path} has stages at chi {[s['chi'] for s in stages]}, not a prefix of {chis}")
    run_kw = {k: o[k] for k in ("tol_E", "tol_grad", "stall_window", "stall_min_evals", "stall_gain")}
    status = stages[-1]["status"] if stages else None
    t0 = time.time()

    def make_chain(k, psi):
        ck = _stage_checkpoint(out, chis, k)
        return var.Chain(psi, model, ops, chis[k], max_sweeps=o["max_sweeps"], checkpoint=ck,
                         log=lambda s: log(f"[chi {chis[k]}] {s}" if len(chis) > 1 else s))

    def load_chain(k):
        """Chain resumed from stage k's checkpoint; Chain.load raises if it was made for another
        model, chi, CTM setting or tensor structure (the template defines the structure)."""
        ck = _stage_checkpoint(out, chis, k)
        template = fpeps.Peps.from_dict(torch.load(ck, weights_only=False, map_location=o["device"])["template"],
                                        config=ops.config)
        D_ck = max(d for s in template.sites() for d in template[s].get_shape()[:4])
        if D_ck != D:
            raise ValueError(f"checkpoint {ck} has D = {D_ck}, not {D}")
        return make_chain(k, template).load(ck)

    # identity check of every existing stage, also of finished ones whose outputs are only rewritten
    for j in range(len(chis)):
        if os.path.exists(_stage_checkpoint(out, chis, j)):
            load_chain(j)
    meta = io.metadata()
    run_info["sessions"].append({**meta, "options": o})  # the settings behind this session's stops
    io.write_json(f"{out}.run.json", run_info)
    if status == "max_steps":  # a step limit is not an outcome: resume that stage (to a larger max_steps)
        stages.pop()
        status = None
    k = len(stages)
    if status is not None and status != "stalled":
        k = len(chis)  # finished earlier; only (re)write the outputs
    while k < len(chis):
        ck = _stage_checkpoint(out, chis, k)
        if os.path.exists(ck):
            chain = load_chain(k)
            log(f"resumed {ck} at step {len(chain.history)}")
        else:
            psi = (seed_state(o["seed"], D, theta, o["sym"], o["noise"], o["rng"], o["cell"], o["schedule"], o["cell_dims"]) if k == 0
                   else io.load_state(_stage_checkpoint(out, chis, k - 1), np_config))
            chain = make_chain(k, var.convert(psi, ops.config))
            chain.save(ck)  # so that a deadline before the first step still leaves a resumable stage
            if k:
                log(f"stage {k}: chi {chis[k - 1]} stalled; continuing at chi {chis[k]}")
        status = chain.run(max_steps=o["max_steps"], deadline=opt_deadline, **run_kw)
        if status == "time":
            break
        stages.append({"chi": chis[k], "status": status, "steps": len(chain.history),
                       "session": len(run_info["sessions"]) - 1,  # index into OUT.run.json sessions
                       "energy_opt": chain.history[-1]["energy"] if chain.history else None})
        io.write_json(stages_path, stages)
        if status != "stalled":
            break
        k += 1

    last = len(stages) - 1 if status != "time" else len(stages)
    psi = io.load_state(_stage_checkpoint(out, chis, last), np_config)
    io.save_state(f"{out}.pkl", psi)
    history = []
    for j in range(last + 1):
        ck = _stage_checkpoint(out, chis, j)
        if os.path.exists(ck):
            history += [{**h, "chi": chis[j]} for h in torch.load(ck, weights_only=False, map_location="cpu")["history"]]
    rec = {"task": "variational", "theta_over_pi": o["theta_over_pi"], "D": D, "sym": o["sym"], "cell": _cell(psi),
           "seed": o["seed"], "seed_provenance": _seed_provenance(o["seed"]), "chi_stages": chis[:last + 1], "chi_opt": chis[last], "status": status, "stages": stages,
           "energy_opt": next((h["energy"] for h in reversed(history) if h["chi"] == chis[last]), None),
           "options": o, "metadata": meta, "init": run_info["init"], "sessions": run_info["sessions"],
           "wall_time": time.time() - t0, "history": history}
    # a run cut at the optimization cutoff validates its current state in the reserved time, as
    # "validation_interim" (extrapolate reads only "validation": the chain is not finished)
    interim = status == "time"
    if o["validate"] and (not interim or o["validate_reserve_hours"]):
        vchis = sorted(set(o["validate_chi"] or [2 * D * D, 3 * D * D, 4 * D * D]))
        done = {}
        for c in vchis:  # one chi at a time, so that the deadline is checked between them
            if deadline is not None and time.time() >= deadline:
                break
            done[c] = var.validate(psi, model, [c])
        if len(done) == len(vchis):
            val = {key: {c: done[c][key][c] for c in vchis} for key in ("energies", "class_energies", "ctm_converged")}
            val.update(c3_spread=done[vchis[-1]]["c3_spread"], order=done[vchis[-1]]["order"])
            rec["validation_interim" if interim else "validation"] = {
                **val, "order": {str(s): v for s, v in val["order"].items()}}
        else:  # an incomplete validation is kept apart: extrapolate reads only "validation"
            rec["validation_skipped"] = ("max_hours reached; run the same config again to validate "
                                         "(set validate_reserve_hours to leave time for it)")
            if done:
                rec["validation_partial"] = {"energies": {c: v["energies"][c] for c, v in done.items()},
                                             "ctm_converged": {c: v["ctm_converged"][c] for c, v in done.items()}}
    io.write_json(f"{out}.json", rec)
    return rec


def _run_provenance(out, o):
    """Contents of OUT.run.json (the caller appends this session and writes it): the options that
    made the initial state and one record per session (each invocation that resumes or rewrites the
    chain: metadata and resolved options; stages name the session that ended them), so that a chain
    resumed across code revisions or settings keeps all of them. A rerun with different initialization options raises: it would
    continue the old chain under the new labels. A chain from before this file existed gets the
    options of the first run that finds it, marked unverified, and later runs are held to them."""
    path = f"{out}.run.json"
    init = json.loads(json.dumps({k: o[k] for k in INIT_KEYS}))  # as stored (tuples become lists)
    if os.path.exists(path) and os.path.exists(f"{out}.pt"):  # no checkpoint: an earlier start failed
        info = io.read_json(path)
        old = {k: info["init"].get(k) for k in INIT_KEYS}  # unverified ones too: the first claim binds
        changed = sorted(k for k in INIT_KEYS if old[k] != init[k])
        if changed:
            raise ValueError(f"{out}: the chain was started with {({k: old[k] for k in changed})}, "
                             f"not {({k: init[k] for k in changed})}; use another out")
    else:
        legacy = os.path.exists(f"{out}.pt")
        info = {"init": {**init, **({"unverified": True} if legacy else {})}, "sessions": []}
    return info


def _cell(psi):
    """Unit cell of a PEPS as run (a saved seed brings its own geometry, whatever ``cell`` says)."""
    g = psi.geometry
    return {"type": type(g).__name__, "dims": list(g.dims), "unique_sites": len(psi.sites()),
            "boundary": getattr(g, "boundary", None)}


# ---- validate / line scan -----------------------------------------------------------------

def run_validate(cfg):
    o = options(cfg, VALIDATE, "validate")
    from . import operators, variational as var
    from .hamiltonian import Model
    psi = io.load_state(o["state"], operators.spin1(o["sym"]).config)
    D = max(d for s in psi.sites() for d in psi[s].get_shape()[:4])
    chis = o["chi"] if isinstance(o["chi"], (list, tuple)) else [o["chi"]]
    val = var.validate(psi, Model(np.pi * o["theta_over_pi"]), [int(c) for c in chis],
                       corner_tol=o["corner_tol"], max_sweeps=o["max_sweeps"])
    rec = {"task": "validate", "state": o["state"], "theta_over_pi": o["theta_over_pi"], "sym": o["sym"], "D": D,
           **val, "order": {str(s): v for s, v in val["order"].items()}, "options": o, "metadata": io.metadata()}
    if o["out"]:
        io.write_json(o["out"], rec)
    return rec


def run_line_scan(cfg):
    o = options(cfg, LINE_SCAN, "line_scan")
    import torch
    from . import operators, variational as var
    from .hamiltonian import Model
    torch.set_num_threads(int(o["threads"]))
    psi = io.load_state(o["state"], operators.spin1(o["sym"]).config)
    ops = var.torch_ops(o["sym"], device=o["device"])
    chain = var.Chain(var.convert(psi, ops.config), Model(np.pi * o["theta_over_pi"]), ops, int(o["chi"]),
                      max_sweeps=o["max_sweeps"])
    rec = {"task": "line_scan", "state": o["state"], "theta_over_pi": o["theta_over_pi"], "sym": o["sym"],
           "chi": int(o["chi"]), **var.line_scan(chain, o["steps"]), "options": o, "metadata": io.metadata()}
    if o["out"]:
        io.write_json(o["out"], rec)
    return rec


# ---- extrapolation ------------------------------------------------------------------------

def _result_point(path, chi):
    """(D, E) from a variational/validate JSON: the validated energy at ``chi`` ('max' = the
    largest validated chi, 'min' = the lowest energy over the validated chi). An energy whose CTM
    did not converge (``ctm_converged`` false) is rejected."""
    r = io.read_json(path)
    v = r.get("validation", r)
    Es = {int(k): e for k, e in (v.get("energies") or {}).items()}
    conv = {int(k): c for k, c in (v.get("ctm_converged") or {}).items()}
    if not Es:
        raise ValueError(f"{path} holds no validated energies")
    if chi not in ("max", "min") and int(chi) not in Es:
        raise ValueError(f"{path} has no energy at chi = {chi} (validated: {sorted(Es)})")
    c = max(Es) if chi == "max" else min(Es, key=Es.get) if chi == "min" else int(chi)
    if conv.get(c) is False:
        raise ValueError(f"{path}: the CTM at chi = {c} did not converge; choose another chi")
    E = Es[c]
    D = r.get("D")
    if D is None:
        raise ValueError(f"{path} does not record D")
    return int(D), float(E), (r.get("theta_over_pi"), r.get("sym"))


def run_extrapolate(cfg):
    o = options(cfg, EXTRAPOLATE, "extrapolate")
    from . import extrapolate as ex
    pts = [(int(D), float(E)) for D, E in (o["points"] or [])]
    files = [_result_point(p, o["chi"]) for p in (o["results"] or [])]
    models = {m for _, _, m in files}
    if len(models) > 1:  # one ladder per (theta, symmetry): SPEC §6 keeps U1xU1 and dense apart
        raise ValueError(f"extrapolate: result files mix (theta_over_pi, sym) = {sorted(models, key=str)}")
    pts += [(D, E) for D, E, _ in files]
    best = {}  # lowest energy per D (several seeds or restarts at one D: the best one counts)
    for D, E in pts:
        best[D] = min(E, best.get(D, E))
    if len(best) < 2:
        raise ValueError("extrapolate: give energies at two or more D (points: [[D, E], ...] and/or results: [json, ...])")
    Ds, Es = sorted(best), [best[D] for D in sorted(best)]
    rec = {"task": "extrapolate", "points": pts, "used": list(zip(Ds, Es)),
           "model": dict(zip(("theta_over_pi", "sym"), models.pop())) if models else None, "all": ex.extrapolate(Ds, Es),
           "subsets": {",".join(map(str, k)): v for k, v in ex.subsets(Ds, Es).items()}}
    if o["reference"] is not None:
        ref, err = (o["reference"] if isinstance(o["reference"], (list, tuple)) else (o["reference"], 0.0))
        a = rec["all"]
        rec["reference"] = {"value": ref, "error": err, "deviation": a["estimate"] - ref,
                            "consistent": bool(abs(a["estimate"] - ref) <= a["error"] + err)}
    if o["out"]:
        io.write_json(o["out"], rec)
    return rec


# ---- scan ---------------------------------------------------------------------------------

def run_scan(cfg):
    o = options(cfg, SCAN, "scan")
    from . import scan
    return scan.run_scan(D=int(o["D"]), n_theta=int(o["n_theta"]), env=o["env"], out=o["out"], workers=int(o["workers"]),
                         schedule=o["schedule"], only_best=o["only_best"], from_dir=o["from_dir"], save_dir=o["save_dir"])


TASKS = {"variational": run_variational, "validate": run_validate, "seed": run_seed, "line_scan": run_line_scan,
         "extrapolate": run_extrapolate, "scan": run_scan}
