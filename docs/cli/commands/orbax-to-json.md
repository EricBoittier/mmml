# `mmml orbax-to-json`

Export Orbax checkpoint to JSON.


## Usage

```bash
mmml orbax-to-json --help
```

## Options

```text
usage: mmml orbax-to-json [-h] -o OUTPUT [--params-key PARAMS_KEY] checkpoint

Convert an Orbax checkpoint to a portable JSON file.

positional arguments:
  checkpoint            Orbax checkpoint directory (epoch-* dir or experiment
                        root)

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output JSON path (e.g. DESdimers_params.json)
  --params-key PARAMS_KEY
                        Key to extract from restored checkpoint dict (default:
                        "params")

CLI: export an Orbax checkpoint to a portable JSON file. Usage: mmml orbax-to-
json path/to/epoch-1985 -o DESdimers_params.json
```



## SpookyPhysNet / SO3LR checkpoints

The JSON written by `evaluate_so3lr_spooky_extxyz.py --output` contains
evaluation metrics; it is **not** a model checkpoint. Export the trained Orbax
epoch separately:

```bash
uv run mmml orbax-to-json \
  ~/mmml/artifacts/spooky_so3lr/epoch-0002 \
  --output ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json
```

The exporter combines the checkpoint's training `config` with its
`model_attributes`. The latter identifies the model as `spooky` and records
constructor values such as `features`, `cutoff`, and `max_padded_atoms`.

Use the resulting JSON anywhere MMML accepts `--checkpoint`. For the JAX-MD
backend (the `cg_jaxmd` path):

```bash
uv run mmml md-system \
  --backend jaxmd \
  --setup free_nvt \
  --checkpoint ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json \
  --composition "RES:1" \
  --template-pdb /path/to/monomer.pdb \
  --temperature 300 \
  --ps 1 \
  --output-dir ~/runs/spooky_epoch2_jaxmd
```

For periodic MD, use a periodic setup such as `pbc_nvt`, provide the normal
box/build inputs, and keep the same JSON checkpoint:

```bash
uv run mmml md-system \
  --backend jaxmd \
  --setup pbc_nvt \
  --checkpoint ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json \
  --composition "RES:20" \
  --template-pdb /path/to/monomer.pdb \
  --box-size 30 \
  --temperature 300 \
  --ps 1 \
  --output-dir ~/runs/spooky_epoch2_pbc
```

Current `md-system` SpookyPhysNet calculator construction uses neutral-singlet
conditioning (`Q=0`, spin multiplicity `1`). Do not use this route for charged
or open-shell systems until those conditioning values are exposed by the MD
CLI.

To check that the portable file restores as SpookyPhysNet before a long run:

```bash
uv run mmml health-check \
  --checkpoint ~/mmml/artifacts/spooky_so3lr/epoch-0002_params.json
```

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)

