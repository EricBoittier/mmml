# PET-OMOL S dynamics of CHARMM `HEME` (PyCHARMM)

`RESI HEME` is the 6-coordinate planar heme in
`setup/charmm/toppar/stream/prot/toppar_all36_prot_heme.str` (charge −2).
It is not a CGenFF residue. A HEME-only build reads `top_all36_prot.rtf` and
`par_all36m_prot.prm`, then streams that file, and generates the segment with
`first none last none` so CHARMM does not apply the protein NTER/CTER patches.

`PRES PHEM` (heme–histidine) and `PRES FHEM` (unliganded) are patches, not
residue names. `--residue HEME` builds the library residue as written.

The stream IC table stores bond lengths as zero, so CHARMM cannot build
coordinates from it. The gas builder places the 73 atoms from the HEME
residue in CHARMM's myoglobin CO test coordinates
(`mmml/data/charmm/heme_mbco_coords.txt`). With
`--charmm-zero-energy-terms vdw,elec,bonded`, the CHARMM MM pre-minimize is
skipped and PET relaxes that structure.

PET-OMOL is the UPET model trained on Open Molecules, which includes
organometallics, so the iron stays in the ML potential. PET-MAD does not.

The model reads the total charge and the spin multiplicity (`2S+1`) from the
structure. They are not inferred from the PSF. `RESI HEME` is charge −2.
The iron in that residue has no axial ligand, so the multiplicity is 3
(intermediate-spin Fe(II)), not the metatomic default of charge 0 and a
singlet. `--counterions SOD` places two sodiums on the carboxylates and the
whole-system charge becomes 0; the multiplicity stays 3.

## 1. Export PET-OMOL S

```bash
uv sync --extra all-cuda12 --extra dev --extra metatomic
uv run --no-sync python -c "from upet import save_upet; save_upet(model='pet-omol', size='s', version='1.0.0', output='pet-omol-s-v1.0.0.pt')"
export PET_OMOL_S_CKPT=$PWD/pet-omol-s-v1.0.0.pt
```

`python` on PATH is `/usr/bin/python` and does not have `upet`. `uv run --no-sync`
uses this checkout's `.venv`, where `upet` 0.3.0 already lists `pet-omol` size
`s` version `1.0.0`. `uv sync --extra metatomic` alone drops the CUDA 12 and
dev extras; the sync line above keeps them.

`size='s'` is OMol S. Larger checkpoints are the same call with `size='m'`
or `size='l'`.

## 2. One heme, vacuum, all-ML PyCHARMM

CHARMM still builds the PSF. The dynamics energy is one metatomic evaluation
of the whole molecule (`whole_system`). MM van der Waals, electrostatics, and
bonded terms are zeroed so they are not added on top of PET.

```bash
export CHARMM_HOME=$PWD/setup/charmm
export CHARMM_LIB_DIR=$CHARMM_HOME/lib
# serial libcharmm: export MMML_NO_CHARMM_MPI=1 MMML_NO_MPI_RERUN=1
# GPU teacher: export MMML_METATOMIC_DEVICE=cuda

uv run --no-sync mmml md-system --backend pycharmm \
  --ml-potential-mode metatomic \
  --metatomic-eval-mode whole_system \
  --checkpoint "$PET_OMOL_S_CKPT" \
  --residue HEME --n-molecules 1 --counterions SOD --builder gas \
  --no-include-mm \
  --charmm-zero-energy-terms vdw,elec,bonded \
  --setup free_nve \
  --mini-nstep 20 --ps-nve 0.05 --dt-fs 0.5 \
  --temperature 300 \
  --output-dir scratch/pet_omol_heme
```

The same settings are in `yaml/heme_nve.yaml`:

```bash
uv run --no-sync mmml md-system --config examples/pet_omol_heme/yaml/heme_nve.yaml --checkpoint "$PET_OMOL_S_CKPT"
```

The log line `Metatomic electronic state: charge=0 spin_multiplicity=3` is
the check that PET saw the neutralized triplet. Without `--counterions` the
same line is `charge=-2`.

Pass: CHARMM `ENER` shows a finite `USER` term, minimization finishes, and the
short NVE does not blow up. This is one cofactor plus two sodiums in vacuum.
Section 4 builds the protein around that cofactor.

## 3. Propionate tails as MM, ghost hydrogens on the cuts

`--mm-region propionates` keeps CBA–O2A and CBD–O2D, and the sodiums, in the
MM region. PET evaluates the porphyrin core (formal charge 0, multiplicity 3)
plus one ghost hydrogen on CAA–CBA and one on CAD–CBD. Drop `bonded` from
`--charmm-zero-energy-terms` so CHARMM still holds the tail and the cut bonds.
The ghost is not an MM particle; its force is split onto the two real atoms.

## 4. Sperm-whale myoglobin with CO

`--residue MBCO` builds one protein. The default coordinates are CHARMM's
crystal CRD `setup/charmm/test/data/mbco_au_q0.crd` (153 residues, the crystal
HSD/HSE/HSP protonation, 73-atom heme, CO, and 337 TIP3). Sulfate in that file
is omitted. `PRES PHEM` bonds His93 NE2 to the iron (`MB 93 HEM 1`) with angle
and dihedral autogeneration off. The protein segment uses NTER/CTER. Heme and
CO use `first none last none`.

`yaml/mbco_nve.yaml` uses the solvated benchmark instead:
`setup/charmm/test/cbenchtest/mbco/mbco4985w.crd`. That file is one protein,
heme, CO, and 4985 TIP3 (17491 atoms) in a cube. The side length is 55.49456 Å,
the value in `mbcodyn.inp` next to the coordinates. `box_size` turns on the
CHARMM crystal. The setup stays `free_nve` (minimize, then 0.05 ps of NVE).
Every histidine in this file is HSD, and residue 122 is ASN, so the coordinate
file's formal charge is +1. A periodic build (`box_size` set) replaces the
TIP3 farthest from the protein with one CLA, and the PSF charge is 0 (17489
atoms). The coordinate span is about 57 Å on a side (diagonal about 98 Å), so
the example sets `--dynamics-max-monomer-extent 120`.

Six-coordinate Fe(II)–CO is a singlet. `--mm-region his93` is the PET region:
heme, CO, and the His93 imidazole, charge −2, multiplicity 1, one ghost
hydrogen on CB–CG. The Fe–NE2 bond stays a real CHARMM bond. Without
`--mm-region`, PET sees the whole system at the PSF charge (+2 for the vacuum
crystal, 0 for this periodic cube). Keep the protein CHARMM
energy: do not pass `--charmm-zero-energy-terms`. `--counterions` is rejected
for MBCO.

The protein plus solvent is one monomer, wider than the 30 Å small-molecule
extent cap. A dense ML–ML exclusion list of this size aborts CHARMM the same
way the isolated heme did, so registration skips it. Bonded terms and charges
on the PET atoms are zeroed. Van der Waals among those atoms stays in CHARMM.

```bash
uv run --no-sync mmml md-system --backend pycharmm \
  --ml-potential-mode metatomic \
  --metatomic-eval-mode whole_system \
  --checkpoint "$PET_OMOL_S_CKPT" \
  --residue MBCO --n-molecules 1 \
  --mbco-crd setup/charmm/test/cbenchtest/mbco/mbco4985w.crd \
  --box-size 55.49456 \
  --mm-region his93 \
  --no-include-mm \
  --dynamics-max-monomer-extent 120 \
  --setup free_nve \
  --mini-nstep 2000 --ps-nve 0.05 --dt-fs 0.5 \
  --temperature 300 \
  --output-dir scratch/pet_omol_mbco_water
```

`yaml/mbco_nve.yaml` is the same run. Pass: the log contains
`MbCO: 17489 atoms, formal charge +0, 1 CLA` and `PHEM MB 93 HEM 1`, then
`Metatomic electronic state: charge=-2 spin_multiplicity=1`, minimization
finishes, and the short NVE stays finite. Fragment evaluation is rejected.
A later launch of the same `--output-dir` reads `nve.res` and continues NVE
with those velocities; the previous `nve.dcd` is renamed to `nve.rescued.1.dcd`.

