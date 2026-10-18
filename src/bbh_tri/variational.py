"""Variational ground-state search: AD through CTM + L-BFGS (docs/ad_workflow.md).

One :class:`Chain` optimizes the unique site tensors of a PEPS for one (model, cell, D, chi). The
optimizer's parameters are the raw site-tensor data; the loss normalizes every tensor inside the
graph, builds the PEPS, converges a CTM environment and returns E/site. The gradient comes from
plain backpropagation through the CTM sweeps (default) or from ``fp_ctmrg``'s implicit fixed-point
backward, which was wrong on a D = 3 state with a kept degenerate CTM pair (docs/yastn_findings.md
U4b). Every forward and backward pass runs inside :func:`yastn_compat.exact_svd_backward` with
``DegenerateSVDWarning`` raised as an error (yastn's own SVD backward gives wrong gradients, U4).

The energy at chi_opt is never the result: :func:`validate` re-measures the state with the numpy
backend at larger chi (docs/ad_workflow.md §4).
"""

import contextlib
import dataclasses
import io
import os
import time
import warnings

import numpy as np
import yastn
import yastn.tn.fpeps as fpeps

from . import lattice, measure, operators
from .yastn_compat import DegenerateSVDWarning, exact_svd_backward, peps_to_dict, zeroed_pairs

# Raised by a loss evaluation that cannot be trusted; the chain perturbs and retries (§6).
RECOVERABLE = (DegenerateSVDWarning, FloatingPointError)


def convert(psi, config):
    """Copy of ``psi`` with tensors in ``config`` (numpy <-> torch), geometry preserved."""
    return fpeps.Peps.from_dict(peps_to_dict(psi), config=config)


def torch_ops(sym="dense", dtype="float64"):
    return operators.spin1(sym, backend="torch", default_dtype=dtype)


def pad(psi, D, noise=1e-3, seed=0):
    """Dense ``psi`` embedded in bond dimension D; new entries are noise * (rms entry) Gaussians.

    Existing entries are kept, so a D-1 (or smaller) state is reproduced up to the noise.
    """
    if psi[psi.sites()[0]].config.sym.SYM_ID != "dense":
        raise NotImplementedError("pad: dense tensors only")
    rng = np.random.default_rng(seed)
    tensors = {}
    for s in psi.sites():
        t = psi[s]
        a = t.to_numpy()
        if max(a.shape[:4]) > D:
            raise ValueError(f"site {s} has bond dimension {a.shape[:4]} > {D}")
        rms = np.sqrt(np.mean(np.abs(a) ** 2))
        b = (noise * rms * rng.standard_normal((D, D, D, D) + a.shape[4:])).astype(np.result_type(a, float))
        b[tuple(slice(0, n) for n in a.shape)] = a
        tensors[s] = yastn.Tensor(config=t.config, s=t.s, n=t.n, dtype="complex128" if np.iscomplexobj(b) else "float64")
        tensors[s].set_block(val=b, Ds=b.shape)
    return fpeps.Peps(psi.geometry, tensors=tensors)


def energy(psi, env, ops, model):
    """(E/site, {class: mean bond energy}) without leaving the autograd graph (cf. measure.energy)."""
    e = measure.class_energies(psi, env, ops, model)
    n = len(psi.geometry.sites())
    total = sum(len(lattice.bonds(psi.geometry, c)) / n * v for c, v in e.items())
    return total.real if hasattr(total, "real") else total, e


class Chain:
    """L-BFGS optimization of the site tensors of ``psi`` (torch config) for ``model``.

    CTM: method '2x2', bond dimension ``chi``, ``corner_tol``, ``max_sweeps``; ``svd_tol`` must be > 0
    and well above roundoff (kept exact zeros give undetectable wrong gradients; a cut among
    roundoff-level values warns as a degenerate pair). ``kept_degenerate``: see
    ``yastn_compat._KEPT_DEGENERATE``. ``ad``: 'unrolled' or 'fp' (see :meth:`loss`). Each
    :meth:`step` is one L-BFGS iteration (strong-Wolfe line search, usually 1-3 loss evaluations).
    Zeroing a degenerate pair has no a-priori error bound, so after a step that zeroed one,
    :meth:`check_gradient` compares AD with central differences (first time, then at most every
    ``fd_check_every`` steps); a relative error above ``fd_rtol`` fails the step.
    """

    def __init__(self, psi, model, ops, chi, corner_tol=1e-9, max_sweeps=500, svd_tol=1e-8,
                 eps_multiplet=None, kept_degenerate="zero", ad="unrolled", history=20, lr=1.0, max_ls=20,
                 fd_check_every=10, fd_rtol=1e-3, checkpoint=None, log=None):
        import torch
        self.torch = torch
        if ad not in ("fp", "unrolled"):
            raise ValueError(ad)
        self.model, self.ops, self.log, self.kept_degenerate, self.ad = model, ops, log, kept_degenerate, ad
        self.template = psi
        self.sites = psi.sites()
        self.params = [psi[s]._data.detach().clone().requires_grad_() for s in self.sites]
        opts_svd = {"policy": "fullrank", "D_total": chi, "tol": svd_tol}
        if eps_multiplet is not None:
            opts_svd["eps_multiplet"] = eps_multiplet
        self.ctm_fwd = {"method": "2x2", "corner_tol": corner_tol, "max_sweeps": max_sweeps,
                        "use_qr": False, "opts_svd": opts_svd}
        # One L-BFGS iteration per step(). torch caps the line search at max_eval - 1 evaluations and
        # max_eval defaults to 5 * max_iter // 4 = 1, which leaves no zoom (t = 0 whenever t = 1 is
        # uphill). torch also stops on |g|_max <= tolerance_grad or g.d > -tolerance_change (absolute,
        # 1e-7, 1e-9); raw-parameter gradients are ~1e-5 near convergence, so run() decides instead.
        self.opt_kwargs = {"lr": lr, "max_iter": 1, "max_eval": max_ls + 1, "history_size": history,
                           "line_search_fn": "strong_wolfe", "tolerance_grad": 0.0, "tolerance_change": 0.0}
        self.optimizer = torch.optim.LBFGS(self.params, **self.opt_kwargs)
        # what a checkpoint must match to be resumed (load raises otherwise)
        self.fd_check_every, self.fd_rtol, self._last_fd_step = fd_check_every, fd_rtol, None
        self.meta = {"model": dataclasses.asdict(model), "chi": chi, "ctm": repr(self.ctm_fwd), "ad": ad,
                     "optimizer": repr(self.opt_kwargs),
                     "kept_degenerate": kept_degenerate, "geometry": repr(peps_to_dict(psi).get("geometry")),
                     "sites": [tuple(s) for s in self.sites], "shapes": [tuple(psi[s].get_shape()) for s in self.sites]}
        self.history = []  # one dict per completed step
        self.checkpoint = checkpoint
        self.n_evals = 0

    # ---- state ------------------------------------------------------------------------
    def peps(self, detach=False):
        """Normalized PEPS built from the current parameters."""
        tensors = {}
        for s, p in zip(self.sites, self.params):
            t = self.template[s].clone()
            p = p.detach() if detach else p
            t._data = p / p.norm()
            tensors[s] = t
        return fpeps.Peps(self.template.geometry, tensors=tensors)

    def numpy_peps(self):
        """Normalized PEPS with the numpy backend (for :func:`validate` and saving)."""
        return convert(self.peps(detach=True), operators.spin1(self.ops.config.sym.SYM_ID).config)

    # ---- loss -------------------------------------------------------------------------
    def loss(self):
        """(E/site, {class: energy}) with the graph attached.

        ``ad='fp'``: implicit differentiation of the CTM fixed point (``fp_ctmrg``; its output is
        silenced). ``ad='unrolled'``: plain backpropagation through every CTM sweep, which needs no
        gauge fixing of the fixed point (memory grows with the number of sweeps).
        """
        from yastn.tn.fpeps.envs.fixed_pt import NoFixedPointError, fp_ctmrg
        psi = self.peps()
        env = fpeps.EnvCTM(psi, init="eye")
        if self.ad == "fp":
            with contextlib.redirect_stdout(io.StringIO()):
                env = fp_ctmrg(env, ctm_opts_fwd=self.ctm_fwd, ctm_opts_fp={"opts_svd": {"policy": "fullrank"}})
        else:
            o = self.ctm_fwd
            info = env.iterate_(opts_svd=o["opts_svd"], method=o["method"], max_sweeps=o["max_sweeps"],
                                corner_tol=o["corner_tol"], use_qr=o["use_qr"])
            if not info.converged:
                raise NoFixedPointError(code=1, message=f"CTM not converged: {info}")
        self.n_evals += 1
        return energy(psi, env, self.ops, self.model)

    def _closure(self):
        last = getattr(self, "_last", None)
        if last is not None and all(self.torch.equal(p, q) for p, q in zip(self.params, last[2])):
            for p, g in zip(self.params, last[3]):  # torch re-evaluates where the last step ended
                p.grad = g.clone()
            return self.torch.tensor(last[0], dtype=self.params[0].dtype)
        self.optimizer.zero_grad()
        E, by_class = self.loss()
        with contextlib.redirect_stdout(io.StringIO()):
            E.backward()
        if not all(self.torch.isfinite(p.grad).all() for p in self.params):
            raise FloatingPointError("non-finite gradient")
        self._last = (E.item(), {c: float(np.real(v.item() if hasattr(v, "item") else v)) for c, v in by_class.items()},
                      [p.detach().clone() for p in self.params], [p.grad.clone() for p in self.params])
        return E

    # ---- optimization -----------------------------------------------------------------
    def step(self):
        """One L-BFGS iteration; returns the log entry. Raises RECOVERABLE / NoFixedPointError."""
        t0, n0, z0 = time.time(), self.n_evals, zeroed_pairs()
        with exact_svd_backward(self.kept_degenerate), warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateSVDWarning)
            self.optimizer.step(self._closure)
            # the parameters are now at the accepted point; reuse the line search's last evaluation
            # if it was made exactly there, else evaluate again
            if not all(self.torch.equal(p, q) for p, q in zip(self.params, self._last[2])):
                self._closure()
        E, by_class, _, grads = self._last
        g = float(self.torch.sqrt(sum((gi.abs() ** 2).sum() for gi in grads)))
        entry = {"step": len(self.history) + 1, "energy": E, "class_energies": by_class,
                 "grad_norm": g, "evals": self.n_evals - n0, "zeroed_pairs": zeroed_pairs() - z0}
        if entry["zeroed_pairs"] and (self._last_fd_step is None
                                      or entry["step"] - self._last_fd_step >= self.fd_check_every):
            ad, fd, noise = self.check_gradient()
            entry["fd_check"] = (ad, fd)
            if abs(ad - fd) > self.fd_rtol * abs(ad) + noise:
                raise DegenerateSVDWarning(f"gradient with zeroed degenerate pairs fails the FD check: "
                                           f"AD {ad:.6e} vs FD {fd:.6e}")
            self._last_fd_step = entry["step"]
        entry["time"] = time.time() - t0
        self.history.append(entry)
        if self.log:
            self.log(f"step {entry['step']} E={entry['energy']:.10f} |g|={g:.2e} evals={entry['evals']} "
                     f"t={entry['time']:.1f}s {entry['class_energies']}")
        if self.checkpoint:
            self.save(self.checkpoint)
        return entry

    def check_gradient(self, rel_eps=1e-4, energy_noise=1e-10):
        """(AD, central FD, FD noise) derivative of the loss along the normalized gradient at the
        current parameters (AD value = |g|); FD noise = energy_noise / step, the FD error from a CTM
        energy converged to ~energy_noise. Uses the gradient of the last evaluation, which
        :meth:`step` leaves at the current point."""
        grads = self._last[3]
        gnorm = self.torch.sqrt(sum((g.abs() ** 2).sum() for g in grads))
        scale = self.torch.sqrt(sum((p.detach().abs() ** 2).sum() for p in self.params))
        eps = rel_eps * scale.item()
        x0 = [p.detach().clone() for p in self.params]
        vals = []
        with exact_svd_backward(self.kept_degenerate), self.torch.no_grad():
            for sgn in (1, -1):
                for p, q, g in zip(self.params, x0, grads):
                    p.copy_(q + sgn * eps * g / gnorm)
                vals.append(self.loss()[0].item())
            for p, q in zip(self.params, x0):
                p.copy_(q)
        return gnorm.item(), (vals[0] - vals[1]) / (2 * eps), energy_noise / eps

    def perturb(self, noise=1e-3, seed=None):
        """Add noise * rms Gaussian noise to every parameter and reset the L-BFGS history (§6)."""
        g = self.torch.Generator().manual_seed(len(self.history) if seed is None else seed)
        with self.torch.no_grad():
            for p in self.params:
                p.add_(noise * p.norm() / p.numel() ** 0.5 * self.torch.randn(p.shape, generator=g, dtype=p.dtype))
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer = self.torch.optim.LBFGS(self.params, **self.opt_kwargs)
        self._last = None

    def run(self, max_steps=100, tol_E=1e-8, window=5, tol_grad=1e-6, max_failures=3):
        """Optimize until E changes by < tol_E over ``window`` steps, |g| < tol_grad, or max_steps.

        A failed step (NoFixedPointError, degenerate SVD gradient, NaN) is retried after
        :meth:`perturb`; after ``max_failures`` consecutive failures the error propagates.
        Returns a status string: 'converged', 'grad', 'max_steps'.
        """
        from yastn.tn.fpeps.envs.fixed_pt import NoFixedPointError
        failures = 0
        while len(self.history) < max_steps:
            snapshot = [p.detach().clone() for p in self.params]
            try:
                entry = self.step()
            except (NoFixedPointError, *RECOVERABLE) as e:
                failures += 1
                if self.log:
                    self.log(f"step {len(self.history) + 1} failed ({type(e).__name__}: {e}); retry {failures}")
                if failures > max_failures:
                    raise
                with self.torch.no_grad():
                    for p, q in zip(self.params, snapshot):
                        p.copy_(q)
                self.perturb(seed=1000 * failures + len(self.history))
                continue
            failures = 0
            if entry["grad_norm"] < tol_grad:
                return "grad"
            Es = [h["energy"] for h in self.history[-window - 1:]]
            if len(Es) > window and max(Es) - min(Es) < tol_E:
                return "converged"
        return "max_steps"

    # ---- checkpoint -------------------------------------------------------------------
    def state_dict(self):
        return {"params": [p.detach().clone() for p in self.params],
                "optimizer": self.optimizer.state_dict(),
                "history": list(self.history),
                "template": peps_to_dict(self.template),
                "n_evals": self.n_evals,
                "last_fd_step": self._last_fd_step,
                "meta": self.meta}

    def save(self, path):
        """Atomic checkpoint: raw parameters, L-BFGS state, history, tensor template."""
        tmp = f"{path}.tmp"
        self.torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path):
        d = self.torch.load(path, weights_only=False)
        if d.get("meta") != self.meta:
            raise ValueError(f"checkpoint {path} was made for {d.get('meta')}, not {self.meta}")
        with self.torch.no_grad():
            for p, q in zip(self.params, d["params"]):
                p.copy_(q)
        self.optimizer.load_state_dict(d["optimizer"])
        self.history = list(d["history"])
        self.n_evals = d["n_evals"]
        self._last_fd_step = d["last_fd_step"]
        return self


def validate(psi, model, chis, corner_tol=1e-10, max_sweeps=500):
    """Re-measure a (numpy) PEPS at each chi in ``chis`` (docs/ad_workflow.md §4).

    Returns {"energies": {chi: E}, "class_energies": {chi: {...}}, "order": {site: (m, |Q|)} at
    the largest chi, "c3_spread": max - min of the h/v/d class energies at the largest chi}.
    """
    ops = operators.spin1(psi[psi.sites()[0]].config.sym.SYM_ID)
    out = {"energies": {}, "class_energies": {}}
    for chi in sorted(chis):  # the diagnostics below use the last (largest) chi
        env = measure.ctm_env(psi, chi, corner_tol=corner_tol, max_sweeps=max_sweeps)
        E, by_class = measure.energy(psi, env, ops, model)
        out["energies"][chi], out["class_energies"][chi] = E, by_class
    nn = [by_class[c] for c in ("h", "v", "d") if c in by_class]
    out["c3_spread"] = max(nn) - min(nn)
    out["order"] = {tuple(s): tuple(map(float, v)) for s, v in measure.order_parameters(psi, env, ops).items()}
    return out
