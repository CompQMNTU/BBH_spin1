"""Plot a theta scan from scripts/m3_theta_scan.py: lowest-energy run per theta, E, m, |Q|.

    uv run --with matplotlib python scripts/m3_plot_scan.py results/m3_scan_D4.jsonl docs/figures/m3_scan_D4.png

Per theta the lowest-energy run wins (``bbh_tri.scan.best_per_theta``: ties within 1e-8 go to
the seeds before the random start).
"""

import json
import sys

import numpy as np

from bbh_tri.scan import best_per_theta, run_params


def main(path, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = best_per_theta(path)
    t = np.array([r["theta_over_pi"] for r in best])
    E = np.array([r["energy"] for r in best])
    m = np.array([np.mean(r["m"]) for r in best])
    Q = np.array([np.mean(r["Q"]) for r in best])
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(t, E, "o-", ms=3)
    tt = np.linspace(0, 2, 400)
    ax[0].plot(tt, 3 * (np.cos(np.pi * tt) + np.sin(np.pi * tt)), "k:", lw=0.8, label="FM product")
    ax[0].set_ylabel("E / site")
    ax[0].legend()
    ax[1].plot(t, m, "o-", ms=3)
    ax[1].set_ylabel("m")
    ax[2].plot(t, Q, "o-", ms=3)
    ax[2].set_ylabel("|Q|")
    ax[2].set_xlabel("θ / π")
    for a in ax:
        for x in (0.25, 0.5, 1.25, 1.873):
            a.axvline(x, color="gray", lw=0.5, ls="--")
    (D, env, warm), = {(r["D"], r["env"], run_params(r)["warm_start"]) for r in best}
    with open(path) as f:
        n_starts = len({r["init"] for r in map(json.loads, f)})
    start = f"warm start from {warm}" if warm else f"lowest energy of {n_starts} starts"
    fig.suptitle(f"NN BBH, √3×√3, dense, Env{env}, D = {D} ({start})")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    for r, mi, qi in zip(best, m, Q):
        print(f"{r['theta_over_pi']:.4f} {r['init']:>6} E={r['energy']:+.6f} m={mi:.3f} Q={qi:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
