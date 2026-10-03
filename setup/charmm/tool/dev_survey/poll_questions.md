# CHARMM developer feature poll — proposed questions

One item per high-level feature (from `feature_map.md`). Each is answerable on a
single usage scale. Feed this list to the Google Apps Script that builds the
Form.

**Suggested answer scale (same for every item):**
`I actively USE it` / `I MAINTAIN / develop it` / `I DON'T use it` /
`Didn't know it existed`

**Form intro text (suggested):**
"For each CHARMM feature below, tell us your relationship to it. This helps us
decide what to keep, modernize, document, or retire. 'Maintain' means you touch
its source code; 'Use' means you run it in your science."

---

## QM/MM

1. **QM/MM — SCC-DFTB** (`SCCDFTB`): semiempirical DFTB quantum region for QM/MM.
2. **QM/MM — GAMESS / GAMESS-UK** (`GAMESS`): ab initio QM via the GAMESS packages.
3. **QM/MM — Q-Chem** (`QCHEM`): ab initio QM via Q-Chem.
4. **QM/MM — Gaussian 09** (`G09`): QM/MM via Gaussian 09.
5. **QM/MM — MNDO97** (`MNDO97`): MNDO97 semiempirical QM region.
6. **QM/MM — built-in QUANTUM / SQuanTM / TURBOMOLE / semiempirical**
   (`QUANTUM`, `SQUANTM`, `QTURBO`, `QMMMSEMI`): CHARMM's native and other
   semiempirical QM interfaces.
7. **QM/MM solvent-boundary potentials** (`GSBP`, `SMBP`): GSBP/SMBP boundary
   electrostatics for QM/MM.

## Free energy & enhanced sampling

8. **Free-energy perturbation (PERT / TSM)** (`PERT`, `TSM`, `CHEMPERT`):
   alchemical free-energy calculations.
9. **Adaptive umbrella & reaction-coordinate biasing** (`ADUMB`, `RXNCOR`,
   `RXNCONS`): umbrella sampling and reaction-coordinate restraints.
10. **Replica & path methods** (`REPLICA`, `RPATH`, `PATHINT`, `CPATH`, `TPS`):
    replica/path optimization, path integrals, transition-path sampling.
11. **Distributed replica exchange (REPDSTR)** (`REPDSTR`, `GENCOMM`): replica-
    exchange driver.
12. **Ensemble / ABPO methods** (`ENSEMBLE`): ensemble runs and adaptively-biased
    path optimization.
13. **The string method (STRINGM)** (`STRINGM`): finite-temperature string method.
14. **Multicanonical / Tsallis sampling** (`MULTCAN`, `TSALLIS`): generalized-
    ensemble sampling.
15. **Accelerated / biased sampling (GAMUS, eABF, metadynamics-like, AFM)**
    (`GAMUS`, `EABF`, `DENBIAS`, `HQBM`, `AFM`, `AXD`): biasing and steered methods.
16. **Steered / targeted MD (SMD)** (`SMD`): steered-dynamics pulling restraints.
17. **Monte Carlo & grand-canonical MC** (`MC`, `MEHMC`, `GCMC`, `GENETIC`):
    MC moves, hybrid MC, GCMC, genetic optimization.

## Electrostatics & implicit solvent

18. **Poisson-Boltzmann electrostatics (PBEQ)** (`PBEQ`): finite-difference PB solver.
19. **Generalized-Born implicit solvent** (`GBFIXAT`, `GBINLINE`, `HDGBVDW`,
    `DHDGB`): GB/GBSW/GBMV-family implicit solvent.
20. **Analytic implicit solvent (ACE / FACTS / SASA / SCP-ISM / ASP)**
    (`ACE`, `FACTS`, `SASAE`, `SCPISM`, `ASPENER`, `ASPMEMB`): analytic
    continuum-solvation energies.
21. **Shell / radial solvation models** (`SHELL`, `RDFSOL`): boundary and
    radial-distribution solvation.
22. **RISM integral-equation solvent** (currently unbuilt; see note): RISM
    solvation. *(Keyword not in current build — confirm whether to retire.)*

## Force fields & energy terms

23. **Alternate force fields (MMFF / CFF / OPLS / CGenFF)** (`MMFF`, `CFF`,
    `OPLS`, `CGENFF`): non-default force fields.
24. **Polarizable / advanced electrostatics (CHEQ / FlucQ / PIPF / multipoles)**
    (`CHEQ`, `FLUCQ`, `PIPF`, `MTPL`): charge-equilibration, fluctuating-charge,
    polarizable, and multipole electrostatics.
25. **Constant-pH MD (PHMD)** (`PHMD`): continuous constant-pH dynamics.
26. **Special bonded terms (VALBOND / CMAP / lone pairs / MMPT)** (`VALBOND`,
    `CMAP`, `LONEPAIR`, `MMPT`, `MSMMPT`): hypervalent, CMAP, lone-pair, and
    proton-transfer potentials.
27. **VdW long-range & soft-core (LRVDW / WCA / soft-core / IPS / LJ-PME)**
    (`LRVDW`, `WCA`, `SOFTVDW`, `NBIPS`, `LJPME`): VdW corrections and LJ-PME.
28. **Fast Ewald / PME / FMM backends** (`FASTEW`, `FMA`, `COLFFT`,
    `COLFFT_NOSP`): long-range electrostatics solvers.
29. **Data-driven & NMR-restraint energy terms (GNN / LarmorD / RDC / NOE / SSNMR
    / PMF)** (`GNN`, `LARMORD`, `RDC`, `PNOE`, `SSNMR`, `EPMF`, `ESTATS`):
    ML potentials and experimental-restraint terms.

## Restraints, geometry & manipulation

30. **Geometric / consensus restraints** (`HMCOM`, `RGYCONS`, `DMCONS`,
    `CONSHELIX`, `PROTO`): COM, Rg, distance-matrix, and helix restraints.
31. **Structure overlap / shape / PRIMO** (`OVERLAP`, `SHAPES`, `PRIMO`,
    `PRIMSH`): structure comparison and coarse-grained PRIMO.
32. **Docking & grid potentials** (`DOCK`, `GRID`, `FFTDOCK`): ligand docking and
    grid-based / FFT-accelerated docking.
33. **4D sampling & energy lookup** (`FOURD`, `LOOKUP`): fourth-dimension method
    and tabulated-energy acceleration.
34. **EMAP density-map fitting** (`EMAP`): cryo-EM / density-map restrained dynamics.

## Dynamics, modes & fitting

35. **Integrator & dynamics variants (VV2 / SGLD / PBC / fast-SHAKE / TNPACK /
    MRMD)** (`DYNVV2`, `OLDDYN`, `SGLD`, `PBOUND`, `FSSHK`, `TNPACK`, `MRMD`):
    alternate integrators and dynamics options.
36. **Normal modes & vibrational analysis** (`DIMB`, `MOLVIB`): DIMB normal modes
    and MOLVIB.
37. **Charge / parameter fitting** (`FITCHG`, `FLEXPARM`): charge and flexible-
    parameter fitting.
38. **Reaction-path search (TRAVEL / RMD)** (`TRAVEL`, `RMD`): conjugate-peak /
    restrained reaction-path methods.

## GPU & acceleration

39. **OpenMM offload (+ OpenMM-Torch)** (`OPENMM`, `OMMTORCH`): run dynamics on
    OpenMM, optionally with ML potentials.
40. **BLaDE GPU engine** (`BLADE`): native CHARMM GPU MD engine.
41. **GPU/accelerator backends (CUDA / OpenCL / Metal / GRAPE)** (`CUDA`,
    `OPENCL`, `METAL`, `GRAPE`, `LIBGRAPE`): low-level accelerator paths.
42. **Domain decomposition (DOMDEC, incl. GPU)** (`DOMDEC`, `DOMDEC_GPU`):
    spatial-decomposition parallel engine.

## Build / parallel / platform infrastructure
*(ask these only if polling build maintainers; otherwise consider dropping)*

43. **MPI parallel build** (`MPI`, `PARALLEL`, `PARAFULL`).
44. **FFT / math backend (MKL vs FFTW)** (`MKL`, `FFTW`).
45. **Interactive graphics / no-graphics** (`XDISPLAY`, `NOGRAPHICS`).
46. **Platform / packaging build switches** (`UNIX`, `GNU`, `APPLE`, `STATIC`,
    `CONFIGURE`, `EXPAND`, `LONGLINE`, `NIH`, `SAVEFCM`).

## Miscellaneous / verify-before-poll
*(small or unclear; confirm each is still a real feature before including)*

47. **Misc specialized modules** (`HFB`, `COMP2`, `IMCUBES`, `PM1`, `PMEPLSMA`,
    `TAMD`, `PROTO`): low-level or prototype toggles — recommend confirming
    whether any deserve its own poll line or should be dropped as internal.
