"""Model parameters and bond-class couplings (SPEC §0, §2.3).

H = sum_<ij> a_ij [cos(theta) S_i·S_j + sin(theta) (S_i·S_j)^2]
  + sum_<<ij>> a_ij [J2 S_i·S_j + K2 (S_i·S_j)^2],
with <ij> the three NN classes ('h', 'v', 'd') and <<ij>> the three NNN classes. Each site has one
bond of each class, so E/site = sum over classes of the mean bond energy of that class.

Spatial anisotropy (Niesen & Corboz, PRB 97, 245146, Sec. IV F): a_ij = 1 on the horizontal chains
('h', along a1 = (0, 1)) and a_ij = J_anis on every bond between chains. The paper (NN only) scales
'v' and 'd'; here every NNN bond also connects different chains, so it is scaled too and
J_anis = 0 still means decoupled chains. J_anis = 1 is the isotropic model; the paper uses
0 <= J_anis <= 1.
"""

from dataclasses import dataclass

import numpy as np

from . import lattice, operators


@dataclass(frozen=True)
class Model:
    theta: float
    J2: float = 0.0
    K2: float = 0.0
    J_anis: float = 1.0

    @property
    def has_nnn(self):
        return self.J2 != 0 or self.K2 != 0

    def anisotropy(self, cls):
        """Bond-class prefactor: 1 on the chains ('h'), J_anis on every bond between chains."""
        return 1.0 if cls in lattice.CHAIN_CLASSES else self.J_anis

    def couplings(self):
        """{bond class: (J, K)} for every class with a nonzero coupling (J_anis = 0 drops all but 'h')."""
        base = {c: (np.cos(self.theta), np.sin(self.theta)) for c in lattice.NN_CLASSES}
        base.update({c: (self.J2, self.K2) for c in lattice.NNN_CLASSES})
        out = {}
        for c, (J, K) in base.items():
            a = self.anisotropy(c)
            if a * J != 0 or a * K != 0:
                out[c] = (a * J, a * K)
        return out

    def bond_hamiltonians(self, ops):
        """{bond class: two-site operator J S·S + K (S·S)^2}; classes sharing (J, K) share one tensor."""
        cache = {}
        return {c: cache.setdefault(jk, operators.h_bond(ops, *jk)) for c, jk in self.couplings().items()}


def energy_per_site(class_energies):
    """E/site from mean bond energies per class (one bond per class and site, SPEC §2.3).

    Returns the backend scalar unchanged (no ``float``), so a torch energy keeps its autograd graph
    and can serve as the AD loss; convert only for reporting.
    """
    return sum(class_energies.values())
