"""Imaginary-time evolution driver (SPEC §4.1, §4.2, M3; NNN bonds M5).

Second-order Trotter steps of triangle-MPO gates and routed NNN gates (``gates.model_gates`` at
tau/2, list + reverse), truncated to bond dimension D with a local metric: 'BP' (EnvBP,
simple-update-like) or 'NTU'. NNN gate paths alternate between Trotter steps.
Each tau stage runs until the CTM energy changes by less than ``tol`` between two checks
(``check_every`` steps apart) or ``max_steps`` is reached.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import yastn.tn.fpeps as fpeps

from . import gates, measure

ENVS = {"BP": fpeps.EnvBP, "NTU": fpeps.EnvNTU}
DEFAULT_SCHEDULE = (0.1, 0.05, 0.02, 0.01)
N_VARIANTS = 6  # cycles through every path of each NNN class (2 for 'nnn_d', 3 for the knights)


@dataclass
class Stage:
    tau: float
    steps: int
    converged: bool
    energy: float
    max_truncation_error: float


@dataclass
class EvolveResult:
    energy: float
    class_energies: dict
    stages: list = field(default_factory=list)
    history: list = field(default_factory=list)  # (tau, step, E) at every check
    wall_time: float = 0.0
    timed_out: bool = False  # stopped at ``deadline``; the last stage is unfinished


def normalize_(psi):
    """Scale every site tensor to norm 1. ``evolution_step_`` does not: exp(-tau h) changes the
    norm by ~exp(-tau E) per step, so without this the tensors overflow (E < 0) or underflow
    (E > 0) to NaN after ~10^3 steps. The state and the (relative) truncation are unchanged."""
    for s in psi.sites():
        psi[s] = psi[s] / psi[s].norm()


def default_chi(D):
    return max(D * D, 4)


def evolve(psi, model, ops, D, schedule=DEFAULT_SCHEDULE, env="BP", max_steps=400, check_every=20,
           tol=1e-7, chi=None, svd_tol=1e-12, ctm_tol=1e-9, ctm_sweeps=100, log=None, deadline=None,
           path_variant=None):
    """Evolve ``psi`` in place towards the ground state of ``model``; return an :class:`EvolveResult`.

    ``schedule`` lists the Trotter steps tau; the energy check uses a fresh CTM environment of
    bond dimension ``chi`` (default D^2). ``deadline`` (a ``time.time()`` value): stop after the
    first step that ends past it, with ``timed_out`` set. ``path_variant``: None alternates the
    NNN gate paths between Trotter steps (energy checks then come after whole cycles of
    ``N_VARIANTS`` steps, ``check_every`` rounded up); an int fixes one (``gates.nnn_gates``).
    """
    chi = default_chi(D) if chi is None else chi
    metric = ENVS[env](psi)
    opts_svd = {"D_total": D, "tol": svd_tol}
    # NTU: the default initialization "EAT_SVD" keeps metric-null components after path gates and
    # drifts away from exact states; "EAT" does not (docs/yastn_findings.md, NTU section).
    step_kwargs = {"initialization": "EAT"} if env == "NTU" else {}
    result = EvolveResult(energy=np.nan, class_energies={})
    t0 = time.time()
    E, by_class = np.nan, {}

    def check():
        E, by_class = measure.energy(psi, measure.ctm_env(psi, chi, corner_tol=ctm_tol, max_sweeps=ctm_sweeps), ops, model)
        return E, by_class

    for tau in schedule:
        if path_variant is not None:
            variants = [path_variant]
        else:
            variants = range(N_VARIANTS) if model.has_nnn else [0]
        step_gates = [gates.trotter2(gates.model_gates(psi.geometry, model, ops, tau / 2, v)) for v in variants]
        # alternating paths: check only after whole cycles, or each check sees another path's truncation
        every = -(-check_every // len(step_gates)) * len(step_gates)
        E_old, err, converged, step = None, 0.0, False, 0
        while step < max_steps:
            step += 1
            infos = fpeps.evolution_step_(metric, step_gates[(step - 1) % len(step_gates)], opts_svd=opts_svd,
                                          **step_kwargs)
            normalize_(psi)
            err = max([err] + [i.truncation_error for i in infos])
            if deadline is not None and time.time() >= deadline:
                result.timed_out = True
                break
            if step % every == 0 or step == max_steps:
                E, by_class = check()
                result.history.append((tau, step, E))
                if log:
                    log(f"tau={tau} step={step} E={E:.10f} trunc={err:.2e}")
                if E_old is not None and abs(E - E_old) < tol:
                    converged = True
                    break
                E_old = E
        result.stages.append(Stage(tau, step, converged, E, err))
        if result.timed_out:
            break
    result.energy, result.class_energies = E, by_class
    result.wall_time = time.time() - t0
    return result
