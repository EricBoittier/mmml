# CHARMM feature/module map — poll groupings

Groups the **147 canonical** compile-time keywords (`keywords_canonical.txt`)
into high-level, human-facing features suitable as poll questions. Descriptions
are drawn from the matching `doc/*.info` files (CHARMM's own documentation) and
from `./configure`'s FEATURES table.

Each group lists: human name, one-line description, the KEY_* keywords it
covers, and the doc topic(s) / command(s).

Coverage: every canonical keyword is placed in exactly one group below, except a
short "Ungrouped / unclear" tail. 38 groups.

Conventions: build-infrastructure keywords (UNIX, GNU, APPLE, STATIC, CONFIGURE,
EXPAND, NOGRAPHICS, MPI/PARALLEL/PARAFULL) are grouped as infrastructure, not
science features, but kept so the canonical list is fully accounted for.

---

## QM/MM (quantum mechanics / molecular mechanics)

### 1. QM/MM: SCC-DFTB
Self-consistent-charge density-functional tight-binding semiempirical QM engine
for QM/MM simulations.
- Keywords: `SCCDFTB`, `DFTBMKL`
- Docs: doc/sccdftb.info; topic: SCCDFTB

### 2. QM/MM: GAMESS / GAMESS-UK
Couples CHARMM MM to the GAMESS (US) and GAMESS-UK ab initio QM packages.
- Keywords: `GAMESS`
- Docs: doc/gamess.info, doc/gamess-uk.info; topic: GAMESS

### 3. QM/MM: Q-Chem
Interface to the Q-Chem ab initio quantum chemistry package.
- Keywords: `QCHEM`
- Docs: doc/qchem.info, doc/qmmm.info; topic: QCHEM

### 4. QM/MM: Gaussian (G09)
Interface to Gaussian 09 for QM/MM energies and forces.
- Keywords: `G09`
- Docs: doc/g09.info; topic: G09

### 5. QM/MM: MNDO97
MNDO97 semiempirical QM engine interface.
- Keywords: `MNDO97`
- Docs: doc/mndo97.info; topic: MNDO97

### 6. QM/MM: legacy CHARMM "QUANTUM"/SQUANTM/QTURBO/QMMMSEMI
The built-in QUANTUM semiempirical code, the SQuanTM module, the TURBOMOLE
(QTURBO) interface, and AMBER-style semiempirical QM/MM (QMMMSEMI).
- Keywords: `QUANTUM`, `SQUANTM`, `QTURBO`, `QMMMSEMI`
- Docs: doc/qmmm.info, doc/qturbo.info

### 7. QM/MM solvent-boundary methods (GSBP/SMBP/PBEQ-coupled)
Generalized and smooth solvent boundary potentials for QM/MM electrostatics.
- Keywords: `GSBP`, `SMBP`
- Docs: doc/pbeq.info, doc/qmmm.info

---

## Free-energy, alchemy & sampling

### 8. Free-energy perturbation (PERT / TSM / CHEMPERT)
Thermodynamic-perturbation and thermodynamic-simple-method free-energy
calculations, including chemical perturbation.
- Keywords: `PERT`, `TSM`, `CHEMPERT`
- Docs: doc/pert.info, doc/perturb.info, doc/tsm.info? (see pert.info)

### 9. Adaptive umbrella / reaction-coordinate biasing (ADUMB, RXNCOR, UMBREL)
Adaptive umbrella sampling and general reaction-coordinate definition/biasing.
- Keywords: `ADUMB`, `RXNCOR`, `RXNCONS`
- Docs: doc/adumb.info, doc/umbrel.info, doc/rxncons.info

### 10. Replica & path methods (REPLICA, RPATH, PATHINT, CPATH, TPS)
Replica construction, replica/path optimization, path-integral (quantum
nuclei), conformational paths, and transition-path sampling.
- Keywords: `REPLICA`, `RPATH`, `PATHINT`, `CPATH`, `TPS`
- Docs: doc/replica.info, doc/pimplem.info, doc/tps.info

### 11. Replica exchange / distributed replicas (REPDSTR)
Distributed replica-exchange driver and its inter-replica communication.
- Keywords: `REPDSTR`, `GENCOMM`
- Docs: doc/repdstr.info

### 12. Ensemble & abpo methods
Ensemble-of-replicas runs and adaptively-biased path optimization (ABPO).
- Keywords: `ENSEMBLE`
- Docs: doc/ensemble.info, doc/abpo.info

### 13. The string method (STRINGM)
Finite-temperature string method for minimum-free-energy paths.
- Keywords: `STRINGM`, `MULTICOM`, `NEWBESTFIT`
- Docs: doc/stringm.info

### 14. Multicanonical / Tsallis / multi-canonical sampling
Generalized-ensemble sampling: multicanonical and Tsallis statistics.
- Keywords: `MULTCAN`, `TSALLIS`
- Docs: doc/dynamc.info (generalized ensembles)

### 15. Accelerated/adaptive biasing (GAMUS, eABF, DENBIAS, HQBM, AFM, AXD)
Collection of biased-sampling and steered methods: Gaussian-mixture umbrella
sampling, extended-system adaptive biasing force, density biasing, history-
dependent biasing (metadynamics-like), atomic-force-microscope pulling, and
accelerated dynamics with constraints.
- Keywords: `GAMUS`, `EABF`, `DENBIAS`, `HQBM`, `AFM`, `AXD`
- Docs: doc/gamus.info, doc/eabf.info, doc/denbias.info, doc/hqbm.info,
  doc/afm.info, doc/axd.info

### 16. Steered / targeted / pulling MD (SMD, TMD-adjacent)
Steered molecular dynamics restraints. (Targeted MD `TMD` is undeclared —
see junk report.)
- Keywords: `SMD`
- Docs: doc/mmfp.info (SMD), doc/tmd.info

### 17. Monte Carlo & hybrid MC (MC, MEHMC, GCMC, GENETIC)
Monte Carlo move engine, momentum-enhanced hybrid MC, grand-canonical MC, and
genetic-algorithm optimization.
- Keywords: `MC`, `MEHMC`, `GCMC`, `GENETIC`
- Docs: doc/mc.info, doc/galgor.info

---

## Electrostatics & implicit solvent

### 18. Poisson-Boltzmann / continuum electrostatics (PBEQ)
Finite-difference Poisson-Boltzmann solver for continuum-solvent
electrostatics.
- Keywords: `PBEQ`
- Docs: doc/pbeq.info

### 19. Generalized Born / GB implicit solvent (GBxx, HDGBVDW)
Generalized-Born implicit-solvent variants and heterogeneous-dielectric GB.
- Keywords: `GBFIXAT`, `GBINLINE`, `HDGBVDW`, `DHDGB`
- Docs: doc/gbmv.info, doc/gbsw.info, doc/gbim.info, doc/genborn.info

### 20. Analytic/empirical implicit solvent (ACE, FACTS, SASA, SCPISM, EEF1)
Analytic continuum electrostatics, FACTS, solvent-accessible-surface-area
energy, and SCP-ISM self-consistent implicit solvent.
- Keywords: `ACE`, `FACTS`, `SASAE`, `SCPISM`, `ASPENER`, `ASPMEMB`
- Docs: doc/ace.info, doc/facts.info, doc/sasa.info, doc/scpism.info,
  doc/aspenr.info, doc/aspenrmb.info, doc/eef1.info

### 21. Membrane / interface implicit models (SHELL, RDFSOL, CORSOL-adjacent)
Shell/spherical-boundary solvation and radial-distribution solvation models.
- Keywords: `SHELL`, `RDFSOL`
- Docs: doc/shell.info, doc/rdfsol.info, doc/sbound.info

### 22. RISM integral-equation solvent
Reference interaction-site-model solvation. (`RISM` keyword is undeclared in the
modern build — see junk report; feature documented.)
- Keywords: (none canonical)
- Docs: doc/rism.info

---

## Force fields & energy terms

### 23. Alternate force fields (MMFF, CFF, OPLS, CGENFF)
Support for the Merck (MMFF), consistent (CFF), OPLS, and CHARMM-General
(CGenFF) force fields.
- Keywords: `MMFF`, `CFF`, `OPLS`, `CGENFF`
- Docs: doc/mmff.info, doc/mmff_params.info, doc/cff.info, doc/parmfile.info

### 24. Polarizable & advanced electrostatics (CHEQ, FLUCQ, PIPF, MTP/MTPL)
Charge-equilibration, fluctuating-charge, polarizable intermolecular potential
(PIPF), and atomic multipole (MTP/MTPL) electrostatics.
- Keywords: `CHEQ`, `FLUCQ`, `PIPF`, `MTPL`
- Docs: doc/cheq.info, doc/flucq.info, doc/pipf.info, doc/mtp.info,
  doc/mtpl.info, doc/drude.info

### 25. Constant-pH / titration (PHMD, CONSPH)
Continuous constant-pH molecular dynamics.
- Keywords: `PHMD`
- Docs: doc/phmd.info, doc/consph.info

### 26. Special bonded / valence terms (VALBOND, CMAP, LONEPAIR, MMPT, MSMMPT)
Valence-bond hypervalent terms, CMAP correction maps, lone-pair geometry,
and molecular-mechanics proton-transfer potentials.
- Keywords: `VALBOND`, `CMAP`, `LONEPAIR`, `MMPT`, `MSMMPT`
- Docs: doc/valbond.info, doc/lonepair.info, doc/mmpt.info, doc/msmmpt.info

### 27. Lennard-Jones / van-der-Waals & long-range corrections (LRVDW, WCA, SOFTVDW, NBIPS, LJPME)
VdW long-range corrections, Weeks-Chandler-Andersen split, soft-core VdW for
alchemy, isotropic periodic sum, and LJ-PME.
- Keywords: `LRVDW`, `WCA`, `SOFTVDW`, `NBIPS`, `LJPME`
- Docs: doc/nbonds.info, doc/ewald.info

### 28. Fast Ewald / PME electrostatics & FMM (FASTEW, FMA, COLFFT)
Fast Ewald summation, fast-multipole (FMM/FMA), and column-FFT PME backends.
- Keywords: `FASTEW`, `FMA`, `COLFFT`, `COLFFT_NOSP`
- Docs: doc/ewald.info, doc/fmm.info

### 29. User/data-driven & ML energy terms (GNN, LARMORD, RDC, PNOE, SSNMR, EPMF, ESTATS, DCM)
Graph-neural-network potentials, chemical-shift/RDC/NOE NMR restraints, empirical
potentials of mean force, and energy statistics.
- Keywords: `GNN`, `LARMORD`, `RDC`, `PNOE`, `SSNMR`, `EPMF`, `ESTATS`
- Docs: doc/gnn.info, doc/larmord.info, doc/rdc.info, doc/nmr.info,
  doc/ssnmr.info, doc/epmf.info, doc/mlpot.info, doc/dcm.info

---

## Restraints, geometry & manipulation

### 30. Geometric restraints & consensus (HMCOM, RGYCONS, DMCONS, CONSHELIX, PROTO)
Center-of-mass restraints, radius-of-gyration restraint, distance-matrix
restraints, helix constraints, and protein-topology helpers.
- Keywords: `HMCOM`, `RGYCONS`, `DMCONS`, `CONSHELIX`, `PROTO`
- Docs: doc/cons.info, doc/mmfp.info

### 31. Overlap / shape / structure comparison (OVERLAP, SHAPES, PRIMO, PRIMSH)
Structure overlap metrics, shape descriptors, and PRIMO coarse-grained model.
- Keywords: `OVERLAP`, `SHAPES`, `PRIMO`, `PRIMSH`
- Docs: doc/overlap.info, doc/shapes.info, doc/primo.info

### 32. Docking & grid potentials (DOCK, GRID, FFTDOCK)
Ligand docking, grid-based potentials, and FFT-accelerated docking.
- Keywords: `DOCK`, `GRID`, `FFTDOCK`
- Docs: doc/grid.info, doc/fftdock.info, doc/openmm_dock.info

### 33. 4D / dimension-extension & lookup methods (FOURD, LOOKUP)
Fourth-dimension sampling and tabulated-energy lookup acceleration.
- Keywords: `FOURD`, `LOOKUP`
- Docs: doc/fourd.info

### 34. EMAP / map-restrained fitting
Cryo-EM / density-map manipulation and map-restrained dynamics.
- Keywords: `EMAP`
- Docs: doc/emap.info

---

## Dynamics integrators & analysis

### 35. Integrators & dynamics variants (DYNVV2, OLDDYN, SGLD, PBOUND, FSSHK, TNPACK, MRMD)
Velocity-Verlet v2, legacy integrator, self-guided Langevin dynamics, periodic
boundary, fast SHAKE, truncated-Newton minimizer, and multiple-replica
reactive MD.
- Keywords: `DYNVV2`, `OLDDYN`, `SGLD`, `PBOUND`, `FSSHK`, `TNPACK`, `MRMD`
- Docs: doc/dynamc.info, doc/sgld.info, doc/minimiz.info, doc/mrmd.info

### 36. Normal modes & vibrational analysis (DIMB, MOLVIB)
Diagonalization-in-mixed-basis normal modes and molecular-vibration analysis.
- Keywords: `DIMB`, `MOLVIB`
- Docs: doc/molvib.info, doc/vibran.info

### 37. Charge/parameter fitting (FITCHG, FLEXPARM)
Charge fitting and flexible parameter assignment.
- Keywords: `FITCHG`, `FLEXPARM`
- Docs: doc/fitcharge.info, doc/fitparam.info

### 38. Travel / RMD reaction-path search
Conjugate-peak/TRAVEL reaction-path and restrained MD path search.
- Keywords: `TRAVEL`, `RMD`
- Docs: doc/trek.info, doc/cross.info

---

## GPU & acceleration

### 39. OpenMM offload (OPENMM, OMMTORCH)
Run dynamics on the OpenMM GPU engine, optionally with OpenMM-Torch ML
potentials.
- Keywords: `OPENMM`, `OMMTORCH`
- Docs: doc/openmm.info, doc/gpu.info

### 40. BLaDE GPU engine
Native CHARMM GPU MD engine.
- Keywords: `BLADE`
- Docs: doc/blade.info, doc/gpu.info

### 41. CUDA / OpenCL / Metal / GRAPE accelerators
Low-level GPU/accelerator backends and the GRAPE/ExaFMM hardware path.
- Keywords: `CUDA`, `OPENCL`, `METAL`, `GRAPE`, `LIBGRAPE`
- Docs: doc/gpu.info, doc/fmm.info

### 42. Domain decomposition (DOMDEC)
Spatial domain-decomposition parallel engine and its GPU and MMFF variants.
- Keywords: `DOMDEC`, `DOMDEC_GPU`
- Docs: doc/domdec.info

---

## Build / parallel / platform infrastructure

### 43. MPI parallelization
Core MPI parallel build.
- Keywords: `MPI`, `PARALLEL`, `PARAFULL`
- Docs: doc/parallel.info

### 44. FFT / math backends (MKL, FFTW)
Intel-MKL and FFTW FFT/linear-algebra backends.
- Keywords: `MKL`, `FFTW`
- Docs: doc/ewald.info, doc/install.info

### 45. Graphics / display (XDISPLAY, NOGRAPHICS)
X11 interactive graphics, or the no-graphics build.
- Keywords: `XDISPLAY`, `NOGRAPHICS`
- Docs: doc/graphx.info

### 46. Platform / packaging build switches
Toolchain and packaging selectors that gate platform-specific code.
- Keywords: `UNIX`, `GNU`, `APPLE`, `STATIC`, `CONFIGURE`, `EXPAND`,
  `LONGLINE`, `NIH`, `SAVEFCM`
- Docs: doc/install.info, doc/cmake.info, doc/prefx.info

### 47. Miscellaneous specialized modules
Smaller features grouped for completeness.
- Keywords: `HFB` (hidden-force/holonomic), `COMP2` (second comparison set),
  `IMCUBES` (image cube nonbond), `PM1` (PM1/PM6 semiempirical flag),
  `PMEPLSMA` (PME plasma correction), `PNOE` already grouped, `TAMD`
  (self-guided/temperature-accelerated MD)
- Docs: doc/images.info, doc/tamd.info, doc/hqbm.info

---

## Ungrouped / unclear (canonical keywords needing a human eyeball)

- `HFB` — exact feature name uncertain (no dedicated doc/*.info); placed in
  Misc group 47 provisionally.
- `COMP2` — "second comparison coordinate set"; build/util more than a science
  feature; could fold into infrastructure. (doc: corman.info / coordinate
  manipulation.)
- `PM1` — listed canonical but also appears with `/*PM1orPM6*/`; semiempirical
  Hamiltonian selector under QUANTUM rather than a standalone feature.
- `IMCUBES` — nonbond-list image-cube optimization; an internal performance
  toggle, not a user-facing science feature.
- `PMEPLSMA` — PME plasma/neutralizing-background correction; internal Ewald
  detail.
- `TAMD` — temperature-accelerated MD; has doc/tamd.info but is REMOVE_ITEM'd
  under MPI builds, so its availability is conditional — flag for poll wording.
- `PROTO` — "prototype" catch-all; verify it still gates anything meaningful.

All other 140 canonical keywords are placed in groups 1-47 above.
