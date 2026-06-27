#!/usr/bin/env python
"""
Main entry point for MMML CLI commands.

Provides a unified interface for all MMML command-line tools.
"""

import sys
import argparse

from mmml.cli.completion import MMML_COMMANDS, completion_main, try_autocomplete

_DISPATCH_COMMANDS = tuple(c for c in MMML_COMMANDS if c != "completion")


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) > 1 and sys.argv[1] == "completion":
        return completion_main(sys.argv[2:])

    if try_autocomplete():
        return 0

    parser = argparse.ArgumentParser(
        prog='mmml',
        description='MMML: Machine Learning for Molecular Modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  make-res    Generate residue (PDB, PSF, topology) via PyCHARMM/CGENFF
              Use ``mmml make-res --list-residues`` to browse CGENFF residue names.
  make-box    Pack molecules into periodic box (vacuum or solvated)
  build-crystal  Build symmetry-aware crystals with PyXtal (+ optional ASE opt)
  run         MM/ML simulation (ASE + JAX-MD with hybrid calculator)
  md-system   Run mixed-composition MD setups (free/pbc NVE/NVT + pbc NPT + lambda TI)
  lambda-mbar MBAR post-processing for lambda-dynamics runs
  run-pycharmm  Pure CHARMM heating and equilibration (no ML)
  pycharmm-two-residue-sample  Restrained sampling for a two-residue CHARMM system
  xml2npz     Convert Molpro XML files to NPZ format
  train       Train DCMNet or PhysNetJAX models (PhysNet: use physnet-train)
  physnet-train  Train PhysNetJAX EF from NPZ (supports --config YAML)
  evaluate    Evaluate trained models (coming soon)
  validate    Validate NPZ files against schema
  train-joint Joint PhysNet+DCMNet training with PhysNet transfer initialization
  fix-and-split  Fix units and create train/valid/test splits from NPZ data
  pyscf-dft   GPU-accelerated DFT calculations (energy, gradient, hessian, etc.)
  pyscf-mp2   GPU-accelerated MP2 (post-HF) calculations
  pyscf-evaluate  Evaluate geometries (E, F, D, ESP) in batch; optional --EF, --add-random-noise
  verify-esp-alignment  Verify esp-grid alignment in evaluated NPZ (data generation check)
  normal-mode-sample  Sample geometries along vibrational modes
  physnet-md  PhysNet MD sampling (ASE + JAX-MD)
  physnet-evaluate  Evaluate PhysNet checkpoint on NPZ (E, F, dipole metrics + predictions)
  compare-npz     Compare reference vs model NPZ (per-atom/element force plots)
  ef-train    Train EF (electric-field) equivariant model from NPZ splits or single file
  ef-evaluate Evaluate trained EF model (metrics + plots from test NPZ)
  ef-md       MD with trained EF model (ASE or JIT JAX; replicas, field ramp, etc.)
  interpolate-xyz  Interpolate two XYZ structures via Z-matrix; write NPZ trajectory
  unwrap-traj Unwrap periodic trajectories; optionally fast-write XYZ/extxyz
  sample-diverse-xyz  Pick N diverse structures from XYZ(s); write sampled.npz (SOAP)
  gui         Start the molecular viewer GUI
  extract-checkpoint-metrics  Plot and print training metrics from Orbax checkpoints
  orbax-to-json  Export an Orbax checkpoint to a portable JSON file
  orca-server    Persistent JAX server for ORCA external-tool calculations
  orca-client    ORCA ProgExt client that forwards jobs to orca-server
  orca-external  Standalone ORCA external-tool wrapper (no server)

Examples:
  mmml make-res --list-residues
  mmml make-res --res CYBZ
  mmml make-box --res CYBZ --n 50 --side_length 25.0
  mmml build-crystal -m benzene.xyz --spg 14 --z 2 -o crystal.extxyz
  mmml build-crystal -m h2o.xyz --spg 36 --z 4 --optimize --emt -o h2o.cif
  mmml md-system --setup pbc_npt --composition MEOH:5,TIP3:5 --temperature 300 --pressure 1.0
  mmml md-system --setup free_nvt --backend jaxmd --composition ACO:30 --packmol-radius 22 --flat-bottom-radius 20 --temperature 250
  mmml md-system --setup free_nve --backend pycharmm --residue ACO --n-molecules 4 --flat-bottom-radius 20 --ps 0.5
  mmml md-system --setup pycharmm_minimize --composition ACO:2 --mini-nstep 30
  mmml md-system --setup lambda_ti --composition MEOH:2 --couple-residues 1 --lambda-md-mode free_nve --pre-min-steps 50
  mmml lambda-mbar --run-dir artifacts/meoh_dimer_lambda_ti
  mmml xml2npz input.xml -o output.npz
  mmml xml2npz inputs/*.xml -o dataset.npz --validate
  mmml validate dataset.npz
  mmml fix-and-split --efd data.npz --output-dir ./splits
  mmml fix-and-split --efd data.npz --grid grids.npz --output-dir ./splits
  mmml gui                    # data dir defaults to cwd; load files from file browser
  mmml gui --data-dir ./trajectories
  mmml gui --file simulation.npz
  mmml pyscf-dft --mol "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0" --energy
  mmml normal-mode-sample -i out/04_results.h5 -o out/06_sampled.npz --amplitude 0.1
  mmml pyscf-evaluate -i out/06_sampled.npz -o out/07_evaluated.npz
  mmml pyscf-evaluate -i traj.npz -o out.npz --EF --esp
  mmml physnet-md --checkpoint out/ckpts/cybz_physnet --data out/splits/energies_forces_dipoles_train.npz -o out/
  mmml physnet-train --data splits/train.npz --ckpt-dir ./ckpts/run --tag my_run
  mmml physnet-train --config train.yaml
  mmml physnet-evaluate --checkpoint out/ckpts/cybz_physnet --data out/splits/test.npz -o out/physnet_eval
  mmml ef-train --train-npz splits/train.npz --valid-npz splits/valid.npz --output-dir ./ef_run
  mmml ef-evaluate --params ./ef_run/params.json --test-npz splits/test.npz --output-dir ./ef_eval --save-output-npz
  mmml ef-evaluate --params ./ef_run/params.json --data splits/test.npz --output-dir ./ef_eval --output-h5 ./ef_eval/eval_gui.h5 --rot-augment
  mmml ef-md --params ./ef_run/params.json --data splits/train.npz --steps 5000 --output md.traj
  mmml ef-md -b jax --params ./ef_run/params.json --xyz mol.xyz --thermostat langevin --steps 10000
  mmml active-learning -i out/physnet_md/physnet_ase.traj -o md_sampled.npz --max-temp 300
  mmml pyscf-evaluate -i md_sampled.npz -o md_evaluated.npz
  mmml interpolate-xyz reactants.xyz products.xyz -o path.npz --steps 500
  mmml unwrap-traj wrapped.traj -o unwrapped.extxyz --format extxyz --fast
  mmml orbax-to-json mmml/models/physnetjax/ckpts/DESdimers/epoch-1985 -o DESdimers_params.json
  mmml orca-server --checkpoint mmml/models/physnetjax/defaults/hf_json/test-f41c04c0-62e3-4785-9018-351ffdc161c4_epoch-251_portable.json --warmup
  mmml orca-client -b 127.0.0.1:8888 job_EXT.extinp.tmp

For help on a specific command:
  mmml <command> --help

Shell tab completion (bash/zsh/fish):
  pip install 'mmml[cli]'    # or: pip install argcomplete
  eval "$(mmml completion bash)"
  # or: eval "$(register-python-argcomplete mmml)"
        """
    )
    
    parser.add_argument(
        'command',
        choices=list(_DISPATCH_COMMANDS),
        help='Command to run'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments for the command'
    )
    
    args = parser.parse_args()
    
    # Dispatch to appropriate command
    if args.command == 'make-res':
        from .misc import make_res_cli
        sys.argv = ['mmml make-res'] + args.args
        return make_res_cli.main()

    elif args.command == 'make-box':
        from .misc import make_box_cli
        sys.argv = ['mmml make-box'] + args.args
        return make_box_cli.main()

    elif args.command == 'build-crystal':
        from .misc import build_crystal_cli
        sys.argv = ['mmml build-crystal'] + args.args
        return build_crystal_cli.main()

    elif args.command == 'run':
        from .run.run_sim import main as run_sim_main
        sys.argv = ['mmml run'] + args.args
        return run_sim_main()

    elif args.command == 'md-system':
        from .run import md_system
        sys.argv = ['mmml md-system'] + args.args
        return md_system.main()

    elif args.command == 'lambda-mbar':
        from .run import lambda_mbar
        sys.argv = ['mmml lambda-mbar'] + args.args
        return lambda_mbar.main()

    elif args.command == 'run-pycharmm':
        from .run.run_pycharmm import main
        sys.argv = ['mmml run-pycharmm'] + args.args
        return main()

    elif args.command == 'pycharmm-two-residue-sample':
        from .run.pycharmm_two_residue_sample import main
        sys.argv = ['mmml pycharmm-two-residue-sample'] + args.args
        return main()

    elif args.command == 'xml2npz':
        from .misc import xml2npz
        sys.argv = ['mmml xml2npz'] + args.args
        return xml2npz.main()
    
    elif args.command == 'validate':
        from ..data.npz_schema import validate_npz
        if not args.args:
            print("Error: Please provide an NPZ file to validate", file=sys.stderr)
            print("Usage: mmml validate <npz_file>", file=sys.stderr)
            return 1
        
        # Validate each file provided
        all_valid = True
        for npz_file in args.args:
            print(f"\n{'='*60}")
            print(f"Validating: {npz_file}")
            print('='*60)
            is_valid, info = validate_npz(npz_file, verbose=True)
            if not is_valid:
                all_valid = False
        
        return 0 if all_valid else 1
    
    elif args.command == 'train':
        from . import train
        sys.argv = ['mmml train'] + args.args
        return train.main()

    elif args.command == 'physnet-train':
        from .make import make_training
        sys.argv = ['mmml physnet-train'] + args.args
        return make_training.main()

    elif args.command == 'train-joint':
        from .misc import train_joint
        sys.argv = ['mmml train-joint'] + args.args
        return train_joint.main()
    
    elif args.command == 'evaluate':
        from . import evaluate
        sys.argv = ['mmml evaluate'] + args.args
        return evaluate.main()
    elif args.command == 'downstream':
        from . import downstream
        sys.argv = ['mmml downstream'] + args.args
        return downstream.main()
    
    elif args.command == 'fix-and-split':
        from .misc import fix_and_split
        sys.argv = ['mmml fix-and-split'] + args.args
        return fix_and_split.main()
    
    elif args.command == 'pyscf-dft':
        from .misc import pyscf_dft
        sys.argv = ['mmml pyscf-dft'] + args.args
        return pyscf_dft.main()

    elif args.command == 'pyscf-mp2':
        from .misc import pyscf_mp2
        sys.argv = ['mmml pyscf-mp2'] + args.args
        return pyscf_mp2.main()

    elif args.command == 'pyscf-evaluate':
        from .misc import pyscf_evaluate
        sys.argv = ['mmml pyscf-evaluate'] + args.args
        return pyscf_evaluate.main()

    elif args.command == 'verify-esp-alignment':
        from .misc import verify_esp_alignment
        sys.argv = ['mmml verify-esp-alignment'] + args.args
        return verify_esp_alignment.main()

    elif args.command == 'normal-mode-sample':
        from .misc import normal_mode_sample
        sys.argv = ['mmml normal-mode-sample'] + args.args
        return normal_mode_sample.main()

    elif args.command == 'physnet-md':
        from .misc import physnet_md
        sys.argv = ['mmml physnet-md'] + args.args
        return physnet_md.main()

    elif args.command == 'physnet-evaluate':
        from .misc import physnet_evaluate
        sys.argv = ['mmml physnet-evaluate'] + args.args
        return physnet_evaluate.main()

    elif args.command == 'compare-npz':
        from .misc import compare_npz
        sys.argv = ['mmml compare-npz'] + args.args
        return compare_npz.main()

    elif args.command == 'pyscf-evaluate-mp2':
        from .misc import pyscf_evaluate_mp2
        sys.argv = ['mmml pyscf-evaluate-mp2'] + args.args
        return pyscf_evaluate_mp2.main()
    
    elif args.command == 'ef-train':
        from .misc import ef_train
        sys.argv = ['mmml ef-train'] + args.args
        return ef_train.main()

    elif args.command == 'ef-evaluate':
        from .misc import ef_evaluate
        sys.argv = ['mmml ef-evaluate'] + args.args
        return ef_evaluate.main()

    elif args.command == 'ef-md':
        from .misc import ef_md
        sys.argv = ['mmml ef-md'] + args.args
        return ef_md.main()

    elif args.command == 'active-learning':
        from .misc import active_learning
        sys.argv = ['mmml active-learning'] + args.args
        return active_learning.main()

    elif args.command == 'kernel-fit':
        from .misc import kernel_fit
        sys.argv = ['mmml kernel-fit'] + args.args
        return kernel_fit.main()

    elif args.command == 'interpolate-xyz':
        from .misc import interpolate_xyz
        sys.argv = ['mmml interpolate-xyz'] + args.args
        return interpolate_xyz.main()

    elif args.command == 'unwrap-traj':
        from .misc import unwrap_traj
        sys.argv = ['mmml unwrap-traj'] + args.args
        return unwrap_traj.main()

    elif args.command == 'sample-diverse-xyz':
        from mmml.generate.sample import sample_diverse_xyz
        sys.argv = ['mmml sample-diverse-xyz'] + args.args
        return sample_diverse_xyz.main()

    elif args.command == 'gui':
        from . import gui
        sys.argv = ['mmml gui'] + args.args
        return gui.main()

    elif args.command == 'extract-checkpoint-metrics':
        from .misc import extract_checkpoint_metrics
        sys.argv = ['mmml extract-checkpoint-metrics'] + args.args
        return extract_checkpoint_metrics.main()

    elif args.command == 'orbax-to-json':
        from .misc import orbax_to_json_cmd
        sys.argv = ['mmml orbax-to-json'] + args.args
        return orbax_to_json_cmd.main()

    elif args.command == 'orca-server':
        from mmml.interfaces.orca_external.server import main as orca_server_main
        return orca_server_main(args.args or None)

    elif args.command == 'orca-client':
        from mmml.interfaces.orca_external.client import main as orca_client_main
        return orca_client_main(args.args or None)

    elif args.command == 'orca-external':
        from mmml.interfaces.orca_external.runner import main as orca_external_main
        return orca_external_main(args.args or None)
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

