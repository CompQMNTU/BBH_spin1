"""D -> infinity extrapolation of variational energies (paper §III B, SPEC §6 M4).

The estimate is the midpoint of the lowest finite-D energy and the intercept of a linear fit in
1/D; the error bar is half their difference. The finite-D energies are variational upper bounds,
so the lowest one bounds E(D = inf) from above, and the linear fit usually overshoots below it.
"""

import numpy as np


def extrapolate(Ds, energies):
    """Paper estimate from energies at bond dimensions ``Ds`` (at least two distinct D).

    Returns {"estimate", "error", "linear", "lowest", "slope", "residuals"}: ``linear`` is the 1/D
    fit's intercept, ``lowest`` the lowest energy, ``residuals`` E - fit per point (in input order).
    """
    D = np.asarray(Ds, dtype=float)
    E = np.asarray(energies, dtype=float)
    if D.shape != E.shape or len(set(D.tolist())) < 2:
        raise ValueError("need energies at two or more distinct bond dimensions")
    slope, linear = np.polyfit(1 / D, E, 1)
    lowest = float(E.min())
    linear = float(linear)  # plain floats: results go to JSON
    return {"estimate": (linear + lowest) / 2, "error": abs(lowest - linear) / 2, "linear": linear,
            "lowest": lowest, "slope": float(slope), "residuals": (E - (linear + slope / D)).tolist()}


def subsets(Ds, energies, min_points=2):
    """:func:`extrapolate` over every run of consecutive D (sorted) with at least ``min_points``
    points, keyed by the tuple of D used: the spread across subsets shows how much the estimate
    depends on which bond dimensions enter the fit."""
    order = np.argsort(Ds)
    D = [Ds[i] for i in order]
    E = [energies[i] for i in order]
    return {tuple(D[i:j]): extrapolate(D[i:j], E[i:j])
            for i in range(len(D)) for j in range(i + min_points, len(D) + 1)
            if len(set(D[i:j])) >= 2}  # repeated D (several seeds at one D) alone cannot be fitted
