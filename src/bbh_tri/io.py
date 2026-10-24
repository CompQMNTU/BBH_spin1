"""Saving and loading states and results (SPEC §5).

States are pickled ``yastn_compat.peps_to_dict`` dicts (numpy backend); a ``variational.Chain``
checkpoint (``.pt``) can stand in for a state wherever one is read. Result records carry
:func:`metadata` (SPEC §5: git hash, yastn version, ...). JSON files are written atomically.
"""

import datetime
import importlib.metadata
import json
import os
import pickle
import platform
import socket
import subprocess

import numpy as np

from .yastn_compat import peps_to_dict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(root=_ROOT):
    """(commit hash, dirty) of the repository holding this package, or (None, None) outside git."""
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return rev, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def metadata():
    """Provenance of a run: code revision, package versions, host, start time (UTC)."""
    rev, dirty = git_revision()
    return {"git": rev, "git_dirty": dirty, "bbh_tri": _version("bbh-tri"), "yastn": _version("yastn"),
            "torch": _version("torch"), "numpy": np.__version__, "python": platform.python_version(),
            "host": socket.gethostname(), "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "threads": os.environ.get("OMP_NUM_THREADS")}


def save_state(path, psi):
    """Pickle a numpy-backend PEPS (geometry preserved; ``psi.to_dict()`` drops it)."""
    _makedirs(path)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(peps_to_dict(psi), f)
    os.replace(tmp, path)


def load_state(path, config):
    """PEPS in ``config`` from a state pickle or from a ``variational.Chain`` checkpoint (.pt).

    A checkpoint gives its current parameters, each site tensor normalized as in the loss.
    """
    import yastn.tn.fpeps as fpeps
    if path.endswith(".pt"):
        import torch
        d = torch.load(path, weights_only=False, map_location="cpu")
        template = fpeps.Peps.from_dict(d["template"], config=config)
        for s, p in zip(template.sites(), d["params"]):
            a = p.detach().cpu().numpy()
            template[s]._data = template[s].config.backend.to_tensor(a / np.linalg.norm(a), dtype=template[s].yastn_dtype,
                                                                     device=template[s].device)
        return template
    with open(path, "rb") as f:
        return fpeps.Peps.from_dict(pickle.load(f), config=config)


def write_json(path, rec):
    """Atomic JSON write (a crash leaves the old file, never a truncated one)."""
    _makedirs(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _makedirs(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
