# `mmml umbrella-sample`

Batched distance umbrella NVT sampling (PhysNet/SpookyNet).


## Usage

```bash
mmml umbrella-sample --help
```

## Options

```text
usage: mmml umbrella-sample [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                            [--model {physnet,kernnn}]
                            [--engine {packed_ml,hybrid_jaxmd}]
                            [--structure STRUCTURE] [--from-psf FROM_PSF]
                            [--from-pdb FROM_PDB] [--from-crd FROM_CRD]
                            [--composition COMPOSITION] [--box-size BOX_SIZE]
                            [--ml-resnames ML_RESNAMES]
                            [--atom-name-i ATOM_NAME_I]
                            [--atom-name-j ATOM_NAME_J] [--lr-solver LR_SOLVER]
                            [--structure-index STRUCTURE_INDEX]
                            [--seed-mode {stretch,tile,frames}]
                            [--output-dir OUTPUT_DIR] [--atoms ATOMS]
                            [--atoms2 ATOMS2] [--cv-difference CV_DIFFERENCE]
                            [--cv2-difference CV2_DIFFERENCE]
                            [--wall-angle A,V,C,THETA_MIN[,K]]
                            [--wall-min-bond A,B,C,D,R_MAX[,K]]
                            [--wall-sum WALL_SUM] [--targets TARGETS]
                            [--targets-y TARGETS_Y] [--xi-min XI_MIN]
                            [--xi-max XI_MAX] [--n-windows N_WINDOWS]
                            [--yi-min YI_MIN] [--yi-max YI_MAX]
                            [--n-windows-y N_WINDOWS_Y] [--k K_EV_A2]
                            [--ky K_Y_EV_A2] [--move-with MOVE_WITH]
                            [--move-with2 MOVE_WITH2]
                            [--invert-with INVERT_WITH]
                            [--equilibration-steps EQUILIBRATION_STEPS]
                            [--max-seed-force MAX_SEED_FORCE]
                            [--thermostat {langevin,nose-hoover}]
                            [--langevin-gamma LANGEVIN_GAMMA]
                            [--max-window-temp MAX_WINDOW_TEMP_K]
                            [--replica-exchange] [--rex-freq REX_FREQ]
                            [--temperature TEMPERATURE_K]
                            [--timestep TIMESTEP_FS] [--nsteps NSTEPS]
                            [--printfreq PRINTFREQ] [--savefreq SAVEFREQ]
                            [--seed SEED] [--no-ema] [--overwrite]
                            [--write-window-xyz]

Batched distance umbrella sampling with a PhysNet / SpookyNet checkpoint via
JAX-MD NVT Nose-Hoover.

Input & configuration:
  --config CONFIG       YAML/JSON UmbrellaConfig; CLI flags override file values
                        when set
  --checkpoint CHECKPOINT
                        PhysNet / SpookyNet / KerNN checkpoint
  --structure STRUCTURE
                        Starting geometry: XYZ, PDB, or NPZ with R/Z arrays
                        (packed_ml)
  --from-psf FROM_PSF   CHARMM PSF for hybrid_jaxmd (make-box model.psf)
  --from-pdb FROM_PDB   Coordinate PDB for hybrid_jaxmd (make-box model.pdb)
  --from-crd FROM_CRD   Coordinate CRD for hybrid_jaxmd
  --composition COMPOSITION
                        Packmol composition for hybrid_jaxmd smoke (e.g.
                        AMM1:1,CH3CL:1,TIP3:12)
  --structure-index STRUCTURE_INDEX
                        Frame index for multi-frame XYZ/PDB/NPZ (default: 0)

Scientific model:
  --model {physnet,kernnn}
                        ML backend (default: auto-detect KerNN JSON)
  --cv-difference CV_DIFFERENCE
                        CV1 as an antisymmetric stretch xi = r(A,B) - r(C,D),
                        given as A,B,C,D. For the Menshutkin reaction with
                        dataset order Cl,N,C use --cv-difference 2,0,2,1 (xi =
                        r(C-Cl) - r(C-N)), so reactants sit at negative xi and
                        products at positive xi. Overrides --atoms.
  --cv2-difference CV2_DIFFERENCE
                        CV2 as an antisymmetric stretch A,B,C,D; enables 2D
                        umbrella
  --thermostat {langevin,nose-hoover}
                        Packed-batch thermostat (default: langevin). Nose-Hoover
                        shares one chain across windows and can cascade failures
                        when one replica heats.
  --replica-exchange    Enable Hamiltonian replica exchange between neighbor
                        umbrella windows (bias-only Metropolis; even/odd pairs
                        on the 1D/2D grid)
  --temperature TEMPERATURE_K
                        NVT temperature in K (default: 300)

Execution:
  --seed-mode {stretch,tile,frames}
                        Window seeding: stretch CV to each ξ₀ (default), tile
                        reference, or use consecutive frames from --structure
  --equilibration-steps EQUILIBRATION_STEPS
                        Discard this many leading MD steps before recording
                        frames (default: 0). Window seeds are optimised
                        geometries with no kinetic energy, so the start of each
                        run is a heating transient.
  --max-seed-force MAX_SEED_FORCE
                        Abort if any window seed max|F| exceeds this (eV/Å;
                        default: 15)
  --nsteps NSTEPS       NVT steps (default: 1000)
  --seed SEED           PRNG seed (default: 42)

Output & artifacts:
  --output-dir, -o OUTPUT_DIR
                        Directory for snapshots, trajectories, and summary
  --savefreq SAVEFREQ   Snapshot save interval (default: same as printfreq)
  --overwrite           Allow writing into a non-empty output directory
  --write-window-xyz    Write per-window XYZ trajectories with mass-weighted CoM
                        at the origin (slow for large K×N_frames); default off —
                        umbrella_snapshots.npz is enough for MBAR.
                        umbrella_bin_minima.traj (lowest E_ML+W per window) is
                        always written

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --engine {packed_ml,hybrid_jaxmd}
                        packed_ml: vacuum batched all-ML (default).
                        hybrid_jaxmd: solvated mechanical embedding (ML solute +
                        MM solvent)
  --box-size BOX_SIZE   Cubic box edge (Å) for hybrid_jaxmd (or sibling
                        box.json)
  --ml-resnames ML_RESNAMES
                        Comma-separated residue names forming the ML region
                        (default: AMM1,CH3CL)
  --atom-name-i ATOM_NAME_I
                        PSF atom name for CV atom i (hybrid; overrides --atoms
                        first index)
  --atom-name-j ATOM_NAME_J
                        PSF atom name for CV atom j (hybrid; overrides --atoms
                        second index)
  --lr-solver LR_SOLVER
                        MM long-range solver for hybrid_jaxmd (default: mic)
  --atoms ATOMS         0-based atom indices for CV1 distance (I,J)
  --atoms2 ATOMS2       0-based atom indices for CV2 distance (K,L); enables 2D
                        umbrella
  --wall-angle A,V,C,THETA_MIN[,K]
                        Flat-bottom lower bound (degrees) on the A-V-C angle,
                        keeping an SN2 attack in its backside channel. xi =
                        r(C-X) - r(C-N) does not constrain the angle, so once
                        the bond breaks the leaving group can swing round: gas-
                        phase windows beyond xi = +1.3 sampled a mean N-C-Cl
                        angle of 70 deg (chloride hydrogen-bonded to the
                        ammonium) while reaction-region windows stayed at
                        165-173 deg. Those windows did not crash, so without
                        this the corruption is silent. Menshutkin: --wall-angle
                        1,2,0,130,50 . Repeatable.
  --wall-min-bond A,B,C,D,R_MAX[,K]
                        Flat-bottom bound on min(r(A-B), r(C-D)): the
                        transferring group must stay bonded to one partner or
                        the other. Preferred over --wall-sum for a transfer
                        coordinate, because the allowed sum depends on xi (large
                        where one bond is long, small at the transition state)
                        while the shortest bond does not. Measured for NH3 +
                        CH3Cl: min(r(C-Cl), r(C-N)) has max 2.18 A across the
                        whole training set, at every xi. Menshutkin: --wall-min-
                        bond 2,0,2,1,2.25,100 . Repeatable.
  --wall-sum WALL_SUM   Flat-bottom confinement wall on a sum of distances, as
                        A,B,C,D,UPPER[,K]: penalises r(A,B) + r(C,D) above UPPER
                        (A) with force constant K (eV/A^2, default 50). Required
                        for difference-CV umbrella runs on a fitted potential:
                        xi = r(A,B) - r(C,D) is satisfied just as well by a
                        dissociated complex, and the fit is unbounded below out
                        there, so the trajectory escapes. Menshutkin: --wall-sum
                        2,0,2,1,6.5 . Repeatable.
  --targets TARGETS     Comma-separated CV1 centers ξ₀ (Å)
  --targets-y TARGETS_Y
                        Comma-separated CV2 centers η₀ (Å); product grid with
                        --targets
  --xi-min XI_MIN       CV1 grid start (Å) if --targets omitted
  --xi-max XI_MAX       CV1 grid end (Å) if --targets omitted
  --n-windows N_WINDOWS
                        Number of CV1 windows on [xi-min, xi-max]
  --yi-min YI_MIN       CV2 grid start (Å)
  --yi-max YI_MAX       CV2 grid end (Å)
  --n-windows-y N_WINDOWS_Y
                        Number of CV2 windows on [yi-min, yi-max]
  --k K_EV_A2           CV1 harmonic force constant (eV/Å²); shared across
                        windows (default: 10)
  --ky K_Y_EV_A2        CV2 force constant (eV/Å²); default same as --k
  --move-with MOVE_WITH
                        Atoms translated rigidly with CV1 atom_j when seeding
                        (e.g. NH3: --atoms 2,1 --move-with 1,3,4,5 fixes C,
                        moves N+H)
  --move-with2 MOVE_WITH2
                        Atoms translated rigidly with CV2 mobile end when
                        seeding
  --invert-with INVERT_WITH
                        Atoms Walden-blended when seeding a shared-hub 2D
                        stretch (e.g. CH3 hydrogens: --invert-with 6,7,8)
  --langevin-gamma LANGEVIN_GAMMA
                        Langevin friction γ (1/fs in jax-md units; default: 0.1)
  --max-window-temp MAX_WINDOW_TEMP_K
                        Abort if any window kinetic T exceeds this (K; default:
                        5× --temperature)
  --rex-freq REX_FREQ   Attempt RE swaps every this many steps (default: 100)
  --timestep TIMESTEP_FS
                        Timestep in fs (default: 0.1)
  --printfreq PRINTFREQ
                        Print interval in steps (default: 100)
  --no-ema              Prefer non-EMA checkpoint params

CLI for batched umbrella NVT sampling with PhysNet / SpookyNet. Usage: # Fix C
(2), move NH3 rigidly along N–C: mmml umbrella-sample \ --checkpoint
examples/m/kl.json \ --structure examples/m/neb/reag_0_opt.xyz \ --atoms 2,1
--move-with 1,3,4,5 \ --xi-min 1.5 --xi-max 3.5 --n-windows 11 \ --k 20
--timestep 0.1 --temperature 300 --nsteps 5000 \ -o out/umbrella --overwrite #
2D (Cl–C × N–C); invert CH3, avoid 1.5/1.5 corner mmml umbrella-sample
--checkpoint examples/m/kl.json \ --structure examples/m/neb/reag_0_opt.xyz \
--atoms 0,2 --atoms2 1,2 \ --move-with2 1,3,4,5 --invert-with 6,7,8 \ --xi-min
1.8 --xi-max 3.0 --n-windows 4 \ --yi-min 1.8 --yi-max 3.0 --n-windows-y 4 \ --k
10 --ky 10 -o out/umbrella2d --overwrite # NPZ (R, Z) or PDB also work; --seed-
mode frames uses consecutive frames as windows mmml umbrella-sample --checkpoint
ckpt.json --structure data.npz \ --atoms 0,1 --targets 1.8,2.0,2.2 --seed-mode
frames -o out/umb
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
