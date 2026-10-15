"""Model parameters and bond-class couplings (SPEC §0, §2.3).

H = sum_<ij> [cos(theta) S_i·S_j + sin(theta) (S_i·S_j)^2] + sum_<<ij>> [J2 S_i·S_j + K2 (S_i·S_j)^2],
with <ij> the three NN classes ('h', 'v', 'd') and <<ij>> the three NNN classes. Each site has one
bond of each class, so E/site = sum over classes of the mean bond energy of that class.
"""

from dataclasses import dataclass

import numpy as np

from . import lattice, operators


@dataclass(frozen=True)
class Model:
    theta: float
    J2: float = 0.0
    K2: float = 0.0

    @property
    def has_nnn(self):
        return self.J2 != 0 or self.K2 != 0

    def couplings(self):
        """{bond class: (J, K)} for every class with a nonzero coupling."""
        out = {c: (np.cos(self.theta), np.sin(self.theta)) for c in lattice.NN_CLASSES}
        if self.has_nnn:
            out.update({c: (self.J2, self.K2) for c in lattice.NNN_CLASSES})
        return out

    def bond_hamiltonians(self, ops):
        """{bond class: two-site operator J S·S + K (S·S)^2}; classes sharing (J, K) share one tensor."""
        cache = {}
        return {c: cache.setdefault(jk, operators.h_bond(ops, *jk)) for c, jk in self.couplings().items()}


def energy_per_site(class_energies):
    """E/site from mean bond energies per class (one bond per class and site, SPEC §2.3)."""
    return float(sum(class_energies.values()))
