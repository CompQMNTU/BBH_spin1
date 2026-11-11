# Literature for M6 (NNN scans)

The SPEC §7 literature pass, done 2026-11-11. Only papers actually opened are cited. "Opened" means the full text (PDF) where it says so, otherwise the arXiv abstract page.

## Szasz, Wang, He: the J–K–J₂ model by DMRG

A. Szasz, C. Wang, Y.-C. He, *Phase diagram of a bilinear-biquadratic spin-1 model on the triangular lattice from density matrix renormalization group simulations*, arXiv:2206.04087 (v1, 8 Jun 2022). The arXiv page links DOI 10.1103/PhysRevB.106.115103. Full text opened.

**Hamiltonian (their Eq. 1):**

    H = Σ_⟨ij⟩ [J S_i·S_j + K (S_i·S_j)²] + J₂ Σ_⟨⟨ij⟩⟩ S_i·S_j

- It uses plain (S·S)², not a Q·Q form. Their Eq. 7 rewrites it as (S·S)² = Q·Q/2 − S·S/2 + 4/3 only to discuss the SU(3) point.
- They take J, J₂ > 0, K of either sign, and no K₂.
- This is SPEC form (1) with J₂ added. Ours has J = cos θ, K = sin θ and the same J₂ S·S on the NNN bonds (our K₂ = 0).

**Mapping to our parameters** (needs cos θ > 0, i.e. θ ∈ (−π/2, π/2)):

| theirs | ours |
|---|---|
| K/J | tan θ |
| J₂/J | J₂ / cos θ |
| energy per site in units of J | E_ours / cos θ |

- **Check:** their quoted Niesen–Corboz transition, FQ→AFM at K ≈ −0.42 J, is tan(1.873π) = −0.4216, matching SPEC §3.
- **Cylinder:** YC6 with infinite length; every high-symmetry point (Γ, K, M) lies on an allowed momentum cut.
- **Bond dimensions:** maps at χ = 2000; checks at 1000 and 4000 along the cuts J₂/J = 0 and 0.1.
- **Map range:** J₂/J ∈ [0, 0.3], K/J ∈ [−1, 1]. In our variables that is θ ∈ [1.75π, 2π) ∪ [0, π/4] with J₂ ≤ 0.3 cos θ. The AFQ interior θ ∈ (π/4, π/2) lies outside their map.

**Phases found:** 120° AFM, stripe, tetrahedral, FQ, AFQ, and no disordered phase.

- **Stripe** is identified by S(q) large at one M point.
- **Tetrahedral** (four-sublattice, non-coplanar) is identified by nonzero scalar chirality ⟨S·(S×S)⟩ and order at all three M points.
- **FQ** has its Q(q) peak at Γ. Representative point (J₂/J, K/J) = (0.1, −1), which is ours θ = 1.75π, J₂ = 0.0707. In FQ they find two nearly degenerate symmetry-broken MPS.
- **AFQ** has its Q(q) peaks at K. Representative point (0.1, 1.0), which is ours θ = π/4, J₂ = 0.0707.
- **120° AFM** covers part of 0.04 ≲ J₂/J ≲ 0.25 at K = 0, where a Green's-function study (their ref. 94, not opened) predicted a disordered state.

**120° → AFQ transition:**

- **Location.** At J₂ = 0 it is at the SU(3) point K = J (θ = π/4). At J₂/J = 0.1 it is at K ≈ 0.7 J, placed where the longest spin-1 and spin-2 correlation lengths cross. In our variables that is θ ≈ 0.194π, J₂ ≈ 0.082.
- **No first-order transition on YC6.** The MPS ground state evolves continuously across it, and the overlap between adjacent parameter points is close to 1.
- **The 2D model.** They state this does not rule out a weakly first-order transition in 2D.
- **Window without order.** Near the transition there is a window with no symmetry breaking and no order at large χ. The window shifts with χ (1000 → 4000), and it is wider at J₂/J = 0.1.
- **Why they reject a spin liquid there:**
  - flux insertion is 2π-periodic, so no fractionalization;
  - a small chirality term J_χ S·(S×S) causes no transition, then gives a first-order transition to tetrahedral order.

**Large-S expectation**, as quoted by them from Yu & Kivelson, PRB 101, 214404 (2020) (not opened):

- stripe for K < 0 and J₂ > J/8 − 9|K|/16;
- tetrahedral for K > 0 and J₂ > J/8 − 5K/48.

**Consequences for M6 (SPEC §6):**

- **Cut 1** (θ = 0, J₂ ∈ [0, 0.3]) lies on their K = 0 line. Expect 120° → stripe, with no intermediate phase per them. The classical threshold is J₂ = 1/8.
- **Cut 2** (θ ≈ 1.85–1.9π, i.e. K/J ≈ −0.51 to −0.33): their map covers it and finds no intermediate phase between FQ and 120°/stripe.
- **Cut 3** should move. With J₂ > 0 the 120°–AFQ boundary sits at θ < π/4: a cut at J₂ = 0.082 crossing θ ≈ 0.19π is the direct test of their continuous-transition claim, on the 3-sublattice √3×√3 cell (`fp` AD allowed). The SPEC's range π/4 < θ < π/2 is outside their data.
- **Tetrahedral order** needs a 4-sublattice cell, i.e. (2,2), unrolled AD only (SPEC §4.4). Their map places it at K > 0 and large J₂.

## Niesen & Corboz (NN model, reference paper)

I. Niesen, P. Corboz, PRB 97, 245146 (2018), arXiv:1805.00354. This is the SPEC's reference; its numbers are in SPEC §3. Seen on the arXiv abstract page:

- the NN phases are FM, 120°, FQ and AFQ;
- all transitions are first order;
- there is no quantum paramagnetic phase.

## 2022–2026 search: nothing new on the model

I looked for later work on the triangular J–K(–J₂) spin-1 model in two ways:

- the Semantic Scholar citation lists of arXiv:2206.04087 and arXiv:1805.00354, checked 2026-11-11;
- arXiv and web searches.

**Result:** no later ground-state study of the model was found. Szasz et al. remain the only J₂ study. Citing works opened at abstract level that are not about this model:

- Hasik & Corboz, arXiv:2311.05534 (2023): an iPEPS ansatz for incommensurate spirals at a cost independent of wavelength, applied to the spin-½ anisotropic triangular model. It is relevant as a method if an incommensurate phase appears at larger J₂ or with J₃.
- Bapna & Bhattacharjee, arXiv:2305.18995 (2023): elastic signatures (sound velocity, magnetostriction) of the spin-1 triangular bilinear-biquadratic nematic phase; finite-T phenomenology.
- Sutcliffe et al., arXiv:2411.05919 (2024): SU(N) spin–phonon dynamics. Not relevant.

**Caveat:** the search was not exhaustive (one citation index, keyword searches). Recheck before writing up M6 results.

## J₃ and NiGa₂S₄

- **C. Stock et al.**, PRL 105, 037402 (2010), arXiv:1006.3238 (abstract opened).
  - Neutron scattering finds short-range incommensurate order, Q₀ = (0.155, 0.155, 0), static for T > 1.5 K, with Θ_CW = −80 K.
  - It is fitted by an AF third-neighbour J₃ = 2.8(6) meV and a ferromagnetic J₁ = −0.35(9) J₃.
  - So the material's dominant coupling is J₃ (distance 2), not our J₂ (distance √3), with J₁ < 0, which puts it well away from our cuts. A J₃ term would be a new bond class: in the square embedding the distance-2 vectors are (0,2), (2,0) and (−2,2), each crossing two NN steps.
- **P. Rubin, A. Sherman, M. Schreiber**, Acta Phys. Pol. A 126, 242 (2014), arXiv:1406.2826 (abstract opened).
  - The spin-1 J₁–J₃ Heisenberg model with J₁ = −(1−p)J, J₃ = pJ, solved by Mori projection operators (no assumed order).
  - It finds FM → disordered at p ≈ 0.2, then AF long-range order.
  - There is no biquadratic term.
- **M. E. Valentine et al.**, PRL 125, 197201 (2020), arXiv:2005.06073 (abstract opened).
  - Raman finds an E_g phonon that modulates J₁ and can produce quadrupole coupling.
  - It also finds loss of local inversion symmetry on cooling (S vacancies), giving disordered DM interactions that can prevent long-range order.
  - This is a material-side argument for biquadratic terms and for disorder.
- **No 2022–2026 theory of the J–K–J₃ spin-1 triangular model was found.**
  - Szasz et al. cite a semiclassical J–K–J₃ study (Stoudenmire, Trebst, Balents, PRB 79, 214436 (2009), arXiv:0904.4683; not opened, seen only as a search-result summary).
  - They also cite a fermionic-parton study (Liu, Zhou, Ng, PRB 81, 224417 (2010); not opened).
