# `mmml gui`

Molecular viewer GUI.


## Usage

```bash
mmml gui --help
```

## Options

```text
usage: mmml gui [-h] [--data-dir DATA_DIR | --file FILE] [--port PORT]
                [--host HOST] [--dev] [--no-browser]
                [--model-params MODEL_PARAMS] [--model-config MODEL_CONFIG]

Start the MMML molecular viewer server

options:
  -h, --help            show this help message and exit
  --data-dir, -d DATA_DIR
                        Directory containing molecular data files (default:
                        current directory)
  --file, -f FILE       Single molecular file to view (pre-load instead of
                        browsing)
  --port, -p PORT       Port to run the server on (default: 8000)
  --host HOST           Host to bind to (default: 127.0.0.1)
  --dev                 Development mode: only serve API (use npm run dev for
                        frontend)
  --no-browser          Do not open browser automatically
  --model-params MODEL_PARAMS
                        Path to model parameters JSON for hidden-state
                        inspection
  --model-config MODEL_CONFIG
                        Optional path to model config JSON for hidden-state
                        inspection

Examples:
  # Use current directory as data dir; load files from file browser
  mmml gui

  # Serve all molecular files from a specific directory
  mmml gui --data-dir ./trajectories

  # Pre-load a single file
  mmml gui --file simulation.npz

  # Custom port
  mmml gui --data-dir ./data --port 8080

  # Development mode (React dev server handles frontend)
  mmml gui --data-dir ./data --dev

Supported file formats:
  - .npz  : MMML NPZ format (R, Z, E, F, D, etc.)
  - .traj : ASE trajectory files
  - .pdb  : PDB protein/molecule files
```


---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
