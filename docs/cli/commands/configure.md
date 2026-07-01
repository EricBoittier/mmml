# `mmml configure`

Interactive config / Snakemake wizard.


Interactive wizard for `md-system` YAML, Snakemake scaffolds, and bundled
`cpu_tests` presets.

```bash
mmml configure
mmml configure --list-presets
mmml configure --non-interactive
```

## Usage

```bash
mmml configure --help
```

## Options

```text
usage: mmml configure [-h] [--non-interactive] [--list-presets]
                      [--preset {cpu-spatial-mpi-mini,cpu-dense-liquid-prep,cpu-md-benchmark,cpu-heat-scaling-smoke,cpu-nve-cutoff-sweep,qm-physnet-pipeline}]
                      [-o OUTPUT_DIR]
                      [--workflow {md-single,md-campaign,physnet-train,snakemake-md,preset-menu}]

Interactive, multiple-choice setup for md-system YAML, PhysNet training
configs, Snakemake workflow scaffolding, and bundled cpu_tests presets.

options:
  -h, --help            show this help message and exit
  --non-interactive     Print menu only (for tests); do not read stdin
  --list-presets        List bundled presets (cpu_tests / tutorial layouts)
                        and exit
  --preset {cpu-spatial-mpi-mini,cpu-dense-liquid-prep,cpu-md-benchmark,cpu-heat-scaling-smoke,cpu-nve-cutoff-sweep,qm-physnet-pipeline}
                        Apply a bundled preset non-interactively (see --list-
                        presets)
  -o, --output-dir OUTPUT_DIR
                        Directory to write generated files (default: current
                        directory)
  --workflow {md-single,md-campaign,physnet-train,snakemake-md,preset-menu}
                        Skip menu and run a specific wizard (still prompts for
                        details unless --preset)
```


## Related docs

- [md-system YAML configs](../../md-system-configs.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
