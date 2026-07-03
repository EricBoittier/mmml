# `mmml plot-restart-velocities`

Plot |v| distributions and outliers from CHARMM .res files.


## Usage

```bash
mmml plot-restart-velocities --help
```

## Options

```text
usage: mmml plot-restart-velocities [-h] [--stem STEM]
                                    [--z-threshold Z_THRESHOLD]
                                    [--output OUTPUT] [--dt-ps DT_PS]
                                    [--no-infer-velocities]
                                    [--max-outliers MAX_OUTLIERS] [--quiet]
                                    directory

Plot |v| distributions from CHARMM restart files (heat.NNNN.res) and flag
velocity outliers.

positional arguments:
  directory             Directory containing numbered restarts (e.g.
                        artifacts/md_run2)

options:
  -h, --help            show this help message and exit
  --stem STEM           Restart stem for numbered files (default: heat →
                        heat.0000.res)
  --z-threshold Z_THRESHOLD
                        MAD z-score cutoff for per-atom speed outliers (default:
                        4)
  --output, -o OUTPUT   Write PNG dashboard (default:
                        <directory>/<stem>_velocity_dashboard.png)
  --dt-ps DT_PS         Timestep in ps for coord-delta velocity inference
                        (default: 0.00025 = 0.25 fs)
  --no-infer-velocities
                        Do not infer velocities from consecutive restart
                        coordinates
  --max-outliers MAX_OUTLIERS
                        Max outlier rows in Rich summary (default: 20)
  --quiet
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
