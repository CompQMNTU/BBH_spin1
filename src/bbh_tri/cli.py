"""``bbh-run CONFIG.yaml``: run the task a YAML config describes (tasks: runs.py).

A config is a flat mapping of task options plus ``task``. A ``runs`` list makes it a batch: each
entry is merged over the top-level options, and ``--index i`` (e.g. ``$SLURM_ARRAY_TASK_ID``)
selects one entry; without it the entries run one after another. String options may refer to
other options with ``{name}`` (e.g. ``out: results/m4/pi4_D{D}_{sym}``). ``--set key=value``
overrides an option (the value is parsed as YAML). ``--dry-run`` prints the resolved options.

    uv run --extra ad bbh-run configs/m4_pi4_U1xU1.yaml --index 2
    uv run bbh-run configs/extrapolate_pi4.yaml

Threads: ``threads`` (default 1) is exported to OMP/OPENBLAS/MKL before numpy is imported.
"""

import argparse
import json
import os
import sys


def _yaml_loader():
    """SafeLoader that reads 1e-6 as a float (YAML 1.1, PyYAML's default, needs 1.0e-6; tolerances
    written the usual way would arrive as strings and fail in the comparisons)."""
    import re
    import yaml

    class Loader(yaml.SafeLoader):
        pass
    Loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(
        r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?|[0-9][0-9_]*[eE][-+]?[0-9]+|\.[0-9_]+(?:[eE][-+]?[0-9]+)?"
        r"|\.(?:inf|Inf|INF)|[-+]\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"), list("-+0123456789."))
    return Loader


def load_config(path, overrides=()):
    """List of resolved option dicts (one per run) from a YAML file and ``key=value`` overrides."""
    import yaml
    loader = _yaml_loader()
    with open(path) as f:
        cfg = yaml.load(f, Loader=loader) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a config must be a mapping")
    sets = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        sets[key.strip()] = yaml.load(value, Loader=loader)
    runs = cfg.pop("runs", None)
    entries = [{**cfg, **r, **sets} for r in runs] if runs else [{**cfg, **sets}]  # --set wins over run entries
    for e in entries:
        if "task" not in e:
            raise ValueError(f"{path}: no 'task' (one of variational, validate, seed, line_scan, extrapolate, scan)")
        _resolve_templates(e)
    return entries


def _resolve_templates(e):
    """Fill ``{name}`` references in string options in place, whatever the key order (a template
    may use another template); a reference to an unknown option or a cycle raises."""
    import string
    fields = lambda v: {f for _, f, _, _ in string.Formatter().parse(v) if f} if isinstance(v, str) else set()
    for _ in range(len(e) + 1):
        pending = {k: v for k, v in e.items() if fields(v)}
        if not pending:
            return
        for k, v in pending.items():
            unknown = fields(v) - set(e)
            if unknown:
                raise ValueError(f"option {k!r} refers to unknown option(s) {sorted(unknown)}")
            if not any(fields(e[f]) for f in fields(v)):  # its references are resolved
                e[k] = v.format(**e)
    raise ValueError(f"cyclic templates among {sorted(k for k, v in e.items() if fields(v))}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bbh-run", description=__doc__.splitlines()[0])
    p.add_argument("config")
    p.add_argument("--index", type=int, default=None, help="run only this entry of the config's 'runs' list")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override an option")
    p.add_argument("--dry-run", action="store_true", help="print the resolved options and exit")
    a = p.parse_args(argv)
    try:
        entries = load_config(a.config, a.set)
    except (OSError, ValueError, KeyError) as e:
        p.error(str(e))
    if a.index is not None:
        if not 0 <= a.index < len(entries):
            p.error(f"--index {a.index} out of range: the config has {len(entries)} run(s)")
        entries = [entries[a.index]]
    if a.dry_run:
        print(json.dumps(entries, indent=1))
        return 0
    threads = {str(e.get("threads", 1)) for e in entries}
    if len(threads) > 1:
        p.error(f"runs in one invocation must share 'threads' (BLAS reads it once), got {sorted(threads)}")
    n = threads.pop()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = n

    from .runs import TASKS
    status = 0
    for e in entries:
        task = TASKS.get(e["task"])
        if task is None:
            p.error(f"unknown task {e['task']!r}; one of {sorted(TASKS)}")
        try:
            rec = task(e)
        except ValueError as err:
            print(f"bbh-run: {e['task']}: {err}", file=sys.stderr)
            status = 2
            continue
        print(json.dumps(_summary(rec), default=str), flush=True)
    return status


def _summary(rec):
    """The parts of a record worth printing (histories stay in the files)."""
    return {k: v for k, v in rec.items() if k not in ("history", "options", "metadata", "class_energies", "sessions", "seed_provenance")}


if __name__ == "__main__":
    sys.exit(main())
