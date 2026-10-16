"""Imaginary-time evolution driver for the NN model (SPEC §4.1, §4.2, M3).

Second-order Trotter steps of triangle-MPO gates (``gates.nn_gates`` at tau/2, list + reverse),
truncated to bond dimension D with a local metric: 'BP' (EnvBP, simple-update-like) or 'NTU'.
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


def normalize_(psi):
    """Scale every site tensor to norm 1. ``evolution_step_`` does not: exp(-tau h) changes the
    norm by ~exp(-tau E) per step, so without this the tensors overflow (E < 0) or underflow
    (E > 0) to NaN after ~10^3 steps. The state and the (relative) truncation are unchanged."""
    for s in psi.sites():
        psi[s] = psi[s] / psi[s].norm()


def default_chi(D):
    return max(D * D, 4)


def evolve(psi, model, ops, D, schedule=DEFAULT_SCHEDULE, env="BP", max_steps=400, check_every=20,
           tol=1e-7, chi=None, svd_tol=1e-12, ctm_tol=1e-9, ctm_sweeps=100, log=None):
    """Evolve ``psi`` in place towards the ground state of ``model``; return an :class:`EvolveResult`.

    ``schedule`` lists the Trotter steps tau; the energy check uses a fresh CTM environment of
    bond dimension ``chi`` (default D^2).
    """
    chi = default_chi(D) if chi is None else chi
    metric = ENVS[env](psi)
    opts_svd = {"D_total": D, "tol": svd_tol}
    result = EvolveResult(energy=np.nan, class_energies={})
    t0 = time.time()

    def check():
        E, by_class = measure.energy(psi, measure.ctm_env(psi, chi, corner_tol=ctm_tol, max_sweeps=ctm_sweeps), ops, model)
        return E, by_class

    for tau in schedule:
        step_gates = gates.trotter2(gates.nn_gates(psi.geometry, model, ops, tau / 2))
        E_old, err, converged, step = None, 0.0, False, 0
        while step < max_steps:
            step += 1
            infos = fpeps.evolution_step_(metric, step_gates, opts_svd=opts_svd)
            normalize_(psi)
            err = max([err] + [i.truncation_error for i in infos])
            if step % check_every == 0 or step == max_steps:
                E, by_class = check()
                result.history.append((tau, step, E))
                if log:
                    log(f"tau={tau} step={step} E={E:.10f} trunc={err:.2e}")
                if E_old is not None and abs(E - E_old) < tol:
                    converged = True
                    break
                E_old = E
        result.stages.append(Stage(tau, step, converged, E, err))
    result.energy, result.class_energies = E, by_class
    result.wall_time = time.time() - t0
    return result
