# `mmml neb`

Nudged elastic band (NEB) path sampling with PhysNet.


## Usage

```bash
mmml neb --help
```

## Options

```text
usage: mmml neb [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                [--calculator {physnet,kernnn}] [--initial INITIAL]
                [--final FINAL] [--output-dir OUTPUT_DIR] [--n-images N_IMAGES]
                [--fmax FMAX] [--climb | --no-climb]
                [--interpolate {idpp,linear}] [--optimizer {BFGS,FIRE,MDMin}]
                [--neb-method {improvedtangent,aseneb,eb,spline,string}]
                [--spring-k SPRING_K]
                [--shared-calculator | --no-shared-calculator]
                [--max-steps MAX_STEPS] [--plot | --no-plot] [--overwrite]
                [--pair I,J]

Nudged elastic band (NEB) path sampling with a PhysNet / MMML checkpoint as the
ASE calculator.

Input & configuration:
  --config CONFIG       YAML/JSON NebConfig; CLI flags override file values when
                        set
  --checkpoint CHECKPOINT
                        PhysNet / KerNN / MMML checkpoint

Scientific model:
  --calculator {physnet,kernnn}
                        ASE calculator backend (default: auto-detect from
                        checkpoint)
  --neb-method {improvedtangent,aseneb,eb,spline,string}
                        ASE NEB force method (default: improvedtangent)
  --shared-calculator, --no-shared-calculator
                        Share one ASE calculator across images (default: on)

Execution:
  --max-steps MAX_STEPS
                        Optional optimizer step cap (default: unlimited until
                        fmax)

Output & artifacts:
  --output-dir, -o OUTPUT_DIR
                        Directory for neb.traj / neb.xyz / profile / plot /
                        summary
  --plot, --no-plot     Write neb_plot.png (default: on)
  --overwrite           Overwrite existing artifacts in --output-dir

Diagnostics & safety:
  -h, --help            show this help message and exit

Other options:
  --initial INITIAL     Reactant / initial XYZ
  --final FINAL         Product / final XYZ
  --n-images N_IMAGES   Total band images including endpoints (default: 11)
  --fmax FMAX           Force convergence threshold in eV/Å (default: 0.05)
  --climb, --no-climb   Enable climbing-image NEB (default: off)
  --interpolate {idpp,linear}
                        Band interpolation method (default: idpp)
  --optimizer {BFGS,FIRE,MDMin}
                        Band optimizer (default: BFGS)
  --spring-k SPRING_K   NEB spring constant (default: 0.1)
  --pair I,J            Atom-index pair to log as a distance column
                        (repeatable). Default for 9-atom NH3–CH3Cl: 1,2 (N–C)
                        and 0,2 (Cl–C).

CLI adapter for ASE NEB sampling with an MMML PhysNet checkpoint. Usage: mmml
neb \ --checkpoint examples/m/kl.json \ --initial examples/m/neb/reag_0_opt.xyz
\ --final examples/m/neb/prod_0_opt.xyz \ --output-dir artifacts/nh3_ch3cl/neb \
--n-images 11 --fmax 0.05
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
