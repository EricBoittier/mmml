# `mmml mode-check`

Monomer/cluster FD, X–H stretch, vib, kick (+ PBC FD).


## Usage

```bash
mmml mode-check --help
```

## Options

```text
usage: mmml mode-check [-h] [--pbc-fd] [--composition COMPOSITION]
                       [--checkpoint CHECKPOINT] [--output-dir OUTPUT_DIR]
                       [--output OUTPUT] [--xyz XYZ] [--checks CHECKS]
                       [--include-mm | --no-include-mm]
                       [--include-ml-dimer | --no-include-ml-dimer]
                       [--mm-charge-mode MM_CHARGE_MODE] [--lr-solver LR_SOLVER]
                       [--ewald-omit-self] [--ml-switch-width ML_SWITCH_WIDTH]
                       [--mm-switch-on MM_SWITCH_ON]
                       [--mm-switch-width MM_SWITCH_WIDTH]
                       [--monomer-separation MONOMER_SEPARATION] [--far]
                       [--cutoff-sweep] [--fd-atoms FD_ATOMS] [--fd-dx FD_DX]
                       [--minimize-fmax MINIMIZE_FMAX]
                       [--minimize-steps MINIMIZE_STEPS]
                       [--minimize-freeze-coms | --no-minimize-freeze-coms]
                       [--kick-steps KICK_STEPS] [--kick-delta KICK_DELTA]
                       [--residue RESIDUE] [--n-molecules N_MOLECULES]
                       [--spacing SPACING]
                       [--min-com-start-distance MIN_COM_START_DISTANCE]
                       [--ml-cutoff ML_CUTOFF]
                       [--pbc-mm-switch-on PBC_MM_SWITCH_ON]
                       [--pbc-mm-cutoff PBC_MM_CUTOFF]

Local force / vibrational diagnostics for monomers and small clusters (FD
forces, X–H stretch scans, ASE vibrations, optional kick FFT), plus the PBC
cluster FD check formerly in check_fd.py.

Input & configuration:
  --composition COMPOSITION
                        Residue composition for vacuum checks, e.g. TIP3:1 or
                        TIP3:2
  --checkpoint CHECKPOINT
                        PhysNet / Spooky portable JSON or Orbax checkpoint
                        ($MMML_CKPT / bundled)
  --residue RESIDUE     Residue for --pbc-fd

Scientific model:
  --mm-charge-mode MM_CHARGE_MODE
  --ewald-omit-self     With --lr-solver ewald (typically --pbc-fd): use the
                        MIC/non-Ewald-trained compatibility operator
                        (cross-monomer Ewald only; omit intramolecular and
                        Gaussian self terms). Default full-box Ewald retains
                        both for Ewald-trained models.
  --mm-switch-on MM_SWITCH_ON
  --cutoff-sweep        For n≥2: run mode-check at every COM station on the
                        hybrid handoff ruler (pure ML, handoff mid,
                        mm_switch_on, MM tail, beyond). Writes
                        cutoff_sweep_summary.json. Conflicts with --far /
                        --monomer-separation. Default checks: minimize,fd,bond-
                        scan with per-monomer COM frozen.
  --min-com-start-distance MIN_COM_START_DISTANCE
  --ml-cutoff ML_CUTOFF
                        ML switch width for --pbc-fd (check_fd default: 0.1)
  --pbc-mm-switch-on PBC_MM_SWITCH_ON
                        MM switch-on for --pbc-fd (check_fd default: 7.0)
  --pbc-mm-cutoff PBC_MM_CUTOFF
                        MM switch width for --pbc-fd (check_fd default: 5.0)

Execution:
  --ml-switch-width ML_SWITCH_WIDTH
  --mm-switch-width MM_SWITCH_WIDTH
  --minimize-steps MINIMIZE_STEPS
  --kick-steps KICK_STEPS

Output & artifacts:
  --output-dir OUTPUT_DIR
                        Directory for vacuum mode-check artifacts (required
                        unless --pbc-fd)
  --output OUTPUT       JSON path for --pbc-fd results

Diagnostics & safety:
  -h, --help            show this help message and exit
  --checks CHECKS       Comma-separated: minimize,fd,bond-scan,vibrations,kick
                        (default: minimize,fd,bond-scan,vibrations; for
                        --cutoff-sweep: fd,bond-scan so COM does not drift)

Other options:
  --pbc-fd              Run the PBC residue-cluster analytic vs FD force check
                        (legacy check_fd.py). Ignores vacuum local-mode flags.
  --xyz XYZ             Optional geometry (XYZ/PDB); otherwise build from named
                        monomers
  --include-mm, --no-include-mm
                        Enable hybrid MM (auto-disabled for single monomer;
                        default: true)
  --include-ml-dimer, --no-include-ml-dimer
                        Enable ML dimer term (default: on when n_monomers>=2)
  --lr-solver LR_SOLVER
                        Long-range Coulomb backend (default: mic). Vacuum local
                        checks should stay on mic; use ewald only with --pbc-fd
                        (hybrid-native full-box Ewald, train↔MD path). Not the
                        same as jax_pme --jax-pme-method ewald.
  --monomer-separation MONOMER_SEPARATION
                        COM spacing between monomers in Å (default: 15 for n≥2,
                        2.8 unused for monomers; pass 2.8 only with an oriented
                        --xyz)
  --far                 Place monomers at 15 Å COM separation (default for n≥2;
                        beyond MM handoff for numerical / monomer-parity);
                        conflicts with --monomer-separation / --cutoff-sweep
  --fd-atoms FD_ATOMS
  --fd-dx FD_DX
  --minimize-fmax MINIMIZE_FMAX
  --minimize-freeze-coms, --no-minimize-freeze-coms
                        During minimize, freeze each monomer COM (intramolecular
                        relax only). Default: on for --cutoff-sweep, off
                        otherwise.
  --kick-delta KICK_DELTA
  --n-molecules N_MOLECULES
  --spacing SPACING
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
