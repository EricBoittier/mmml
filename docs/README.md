# MMML package architecture

Overview of the **mmml** Python package: layout, import dependencies, CLI wiring, and MD runtime paths.  
Generated from static analysis (AST import scan) of `mmml/` (~291 Python modules), plus vendored `pycharmm/` and deprecated `scripts/` shims.

---

## Top-level layout

```mermaid
flowchart TB
  subgraph entry["Entry points"]
    MMML["mmml\n(cli/__main__.py)"]
    SCRIPTS["scripts/*.py\n(shims → cli)"]
    PKG["pyproject scripts\nmmml-pycharmm-…, spectra-md"]
  end

  subgraph mmml_pkg["mmml package"]
    CLI["cli\n71 modules"]
    IFACE["interfaces\n61 modules"]
    MODELS["models\n106 modules"]
    UTILS["utils\n26 modules"]
    DATA["data\n12 modules"]
    GEN["generate\n13 modules"]
    GUI["gui\n12 modules"]
    SPEC["spectra\n2 modules"]
  end

  subgraph vendor["Bundled / external"]
    PYCH["pycharmm/\n35 modules"]
    EXT["numpy, jax, ase, pycharmm,\npyscf, flax, optax, …"]
  end

  MMML --> CLI
  SCRIPTS --> CLI
  PKG --> CLI
  PKG --> SPEC

  CLI --> IFACE
  CLI --> MODELS
  CLI --> DATA
  CLI --> UTILS
  CLI --> GEN
  CLI --> GUI

  IFACE --> MODELS
  IFACE --> UTILS
  IFACE --> PYCH
  MODELS --> UTILS
  MODELS --> DATA
  UTILS --> PYCH
  GEN --> IFACE

  CLI --> EXT
  IFACE --> EXT
  MODELS --> EXT
```

| Subpackage | `.py` files (approx.) | Role |
|------------|----------------------:|------|
| `mmml.cli` | 71 | `mmml` dispatcher, MD runners, training/eval/misc CLIs |
| `mmml.models` | 106 | PhysNetJAX, DCMNet, EF models |
| `mmml.interfaces` | 61 | CHARMM, PySCF, ASE, OpenMM, … bridges |
| `mmml.utils` | 26 | I/O, checkpoints, geometry, plotting, hybrid optimization |
| `mmml.data` | 12 | NPZ schema, units, loaders, adapters |
| `mmml.generate` | 13 | Dimers, DMC, sampling utilities |
| `mmml.gui` | 12 | FastAPI viewer + frontend assets |
| `mmml.spectra` | 2 | Spectra MD (`mmml-spectra-md` console script) |

---

## Inter-package imports

Arrow labels are approximate import-statement counts between subpackages (`mmml.cli` → `mmml.interfaces`, etc.).

```mermaid
flowchart LR
  CLI["mmml.cli"]
  IFACE["mmml.interfaces"]
  MODELS["mmml.models"]
  DATA["mmml.data"]
  UTILS["mmml.utils"]
  GEN["mmml.generate"]
  SPEC["mmml.spectra"]

  CLI -->|"57"| IFACE
  CLI -->|"50"| MODELS
  CLI -->|"24"| DATA
  CLI -->|"10"| UTILS
  CLI -->|"1"| GEN

  IFACE -->|"11"| MODELS
  IFACE -->|"2"| UTILS
  IFACE -->|"1"| DATA

  MODELS -->|"7"| UTILS
  MODELS -->|"3"| DATA

  UTILS -->|"3"| MODELS
  UTILS -->|"1"| DATA
  UTILS -->|"1"| IFACE

  GEN -->|"4"| IFACE
  GEN -->|"2"| MODELS

  SPEC -->|"4"| MODELS
```

**CLI is the hub** — most user-facing flows go through `mmml.cli` into `interfaces` (CHARMM/MM) and `models` (ML).

---

## Legacy import aliases

`mmml/__init__.py` registers compatibility names in `sys.modules`:

```mermaid
flowchart LR
  ROOT["mmml"]
  ROOT --> PI["mmml.pycharmmInterface\n(sys.modules alias)"]
  ROOT --> DC["mmml.dcmnet\n(alias)"]
  PI --> IFACE["mmml.interfaces.pycharmmInterface"]
  DC --> DCM["mmml.models.dcmnet"]
```

Prefer canonical paths:

- `mmml.interfaces.pycharmmInterface.*`
- `mmml.models.dcmnet.*`
- `mmml.models.physnetjax.*`

Some CLI code still imports `mmml.physnetjax.*` (legacy) alongside `mmml.models.physnetjax.*`.

---

## `mmml.cli` structure

```mermaid
flowchart TB
  MAIN["cli/__main__.py\n31 subcommands"]

  subgraph make["cli/make"]
    MKRES["make_res"]
    MKBOX["make_box"]
    MKMIX["make_mixed_box"]
  end

  subgraph run["cli/run"]
    MD_SYS["md_system"]
    RUN_SIM["run_sim"]
    LAM["lambda_dynamics"]
    LAM_J["lambda_jaxmd"]
    LAM_MB["lambda_mbar"]
    PBC["md_pbc_suite/\nase, jaxmd, cluster, check_fd"]
    ASE_R["ase_runner"]
    JAX_R["jaxmd_runner"]
    PY_R["pycharmm_runner"]
    BASE["cli/base.py"]
  end

  subgraph misc["cli/misc\n~40 modules"]
    TJ["train_joint"]
    PN["physnet_md / evaluate"]
    EF["ef_train / evaluate / md"]
    PYSCF["pyscf_*"]
    DATA_MISC["xml2npz, fix_and_split, …"]
  end

  subgraph other_cli["cli/"]
    TRAIN["train/train.py"]
    PLOT["plot/*\nnot in dispatcher"]
    GUI_CLI["gui.py"]
  end

  MAIN --> make
  MAIN --> run
  MAIN --> misc
  MAIN --> TRAIN
  MAIN --> GUI_CLI
  MAIN --> DATA_ROOT["data.npz_schema"]

  make --> IFACE_PY["interfaces.pycharmmInterface"]
  MD_SYS --> PBC
  MD_SYS --> LAM
  LAM --> PBC
  LAM --> LAM_J
  RUN_SIM --> ASE_R
  RUN_SIM --> JAX_R
  RUN_SIM --> PY_R
  RUN_SIM --> BASE
  PBC --> BASE
  PBC --> IFACE_PY
  PBC --> UTILS_GEO["utils.geometry_checks"]
  JAX_R --> IFACE_PY
```

### CLI modules not reached from `mmml` dispatcher

Examples (runnable only via `python -m …` or legacy paths):

- `cli/plot/plot_training.py`, `plot_checkpoint_history.py`
- `cli/misc/opt_mmml.py`, `dynamics.py`, `evaluate_model.py`, `compare_*`, …
- `cli/make/make_training.py`, `make_mixed_box.py` (mixed box: examples / direct `-m`)

---

## MD runtime path

Production MD flows after moving suites from `scripts/` to `mmml.cli.run.md_pbc_suite`:

```mermaid
sequenceDiagram
  participant User
  participant md_system as md_system
  participant ase as md_pbc_suite.ase
  participant jax as md_pbc_suite.jaxmd
  participant cluster as md_pbc_suite.cluster
  participant calc as mmml_calculator
  participant phys as models.physnetjax
  participant charmm as pycharmm

  User->>md_system: mmml md-system
  alt lambda_ti
    md_system->>lambda_dynamics: in-process
    lambda_dynamics->>cluster: PSF / cluster build
    lambda_dynamics->>ase: shared helpers
    lambda_dynamics->>calc: setup_calculator
  else ASE backend
    md_system->>ase: main()
    ase->>cluster: _build_psf_ordered_cluster
    ase->>calc: setup_calculator
    ase->>charmm: minimize / PSF
  else JAX-MD backend
    md_system->>jax: main()
    jax->>ase: cluster helpers
    jax->>calc: setup_calculator
    jax->>jaxmd_runner: set_up_nhc_sim_routine
  end
  calc->>phys: EF model, batches, helper_mlp
  calc->>charmm: MM energy / forces
```

### `md_pbc_suite` internal imports

| Module | Main `mmml` dependencies |
|--------|---------------------------|
| `cluster.py` | `interfaces.pycharmmInterface.import_pycharmm`, `utils.get_Z_from_psf` |
| `ase.py` | `cli.base`, `mmml_calculator`, `geometry_checks`, `cluster` |
| `jaxmd.py` | `cli.base`, `jaxmd_runner`, `mmml_calculator`, re-exports from `ase` |
| `check_fd.py` | `cli.base`, `ase` helpers, `cluster` |

---

## `interfaces.pycharmmInterface` hub

Central integration for hybrid ML/MM:

```mermaid
flowchart TB
  CALC["mmml_calculator.py\nsetup_calculator"]

  CALC --> CU["calculator_utils"]
  CALC --> MMEF["mm_energy_forces"]
  CALC --> MLB["ml_batching"]
  CALC --> CUT["cutoffs"]
  CALC --> PBC["pbc_utils_jax"]
  CALC --> IMP["import_pycharmm"]

  CALC --> PN["models.physnetjax\n.calc.helper_mlp\n.models.model.EF\n.data.batches"]
  CALC --> CKPT["utils.model_checkpoint"]

  SETUP["setupBox / setupRes\npackmol_placement"] --> IMP
  ASE_IF["interfaces.aseInterface"] --> CALC
  CLI_RUN["cli/run/*"] --> CALC
  PBC_ASE["md_pbc_suite.ase"] --> CALC
```

### Other interface subpackages

| Subpackage | Files (approx.) | Typical CLI use |
|------------|----------------:|-------------------|
| `pycharmmInterface` | 17 | Core CHARMM + hybrid calculator |
| `pyscf4gpuInterface` | 11 | `mmml pyscf-dft`, `pyscf-evaluate`, … |
| `dcmInterface` | 12 | `mmml kernel-fit` |
| `aseInterface` | 4 | Legacy ASE calculators |
| `chemcoordInterface` | 1 | `mmml interpolate-xyz` |
| `openmmInterface`, `jaxmdInterface`, … | few | Specialized / optional |

---

## `mmml.models` tree

```mermaid
flowchart TB
  MODELS["mmml.models"]

  MODELS --> PNJ["physnetjax\n~58 modules\ntraining, data, calc, models"]
  MODELS --> DCM["dcmnet\n~34 modules\ntraining, loss, electrostatics"]
  MODELS --> EF["EF\n~13 modules\ntraining, evaluate, ase_md, jax_md"]

  CLI_MISC["cli/misc\ntrain_joint, physnet_*"] --> PNJ
  CLI_MISC --> DCM
  CLI_EF["cli/misc/ef_*"] --> EF
  CALC["mmml_calculator"] --> PNJ
```

---

## Third-party dependencies

Counts are approximate import sites under `mmml/` (stdlib excluded).

| Library | ~sites | Used by |
|---------|-------:|---------|
| **ase** | 243 | MD, CLI, interfaces, models |
| **jax** | 191 | Models, calculators, MD |
| **numpy** | 180 | Widespread |
| **pycharmm** | 75 | CHARMM interface, MD suites |
| **matplotlib** | 57 | Plotting, analysis |
| **e3x** | 43 | Equivariant models |
| **rich** | 35 | CLI output |
| **scipy** | 31 | Data, analysis |
| **pandas** | 28 | Data, MD suite |
| **flax** / **optax** | 19 / 17 | Training |
| **pyscf** / **gpu4pyscf** | 21 / 16 | QC interfaces |
| **orbax** | 14 | Checkpoints |
| **h5py** | 15 | Trajectories, QC |

---

## Repo siblings (outside `mmml/`)

```mermaid
flowchart TB
  REPO["repository root"]

  REPO --> MMML["mmml/"]
  REPO --> PYCH["pycharmm/\nsetuptools package"]
  REPO --> SCR["scripts/\nshims + benchmarks"]
  REPO --> TESTS["tests/"]
  REPO --> EX["examples/"]

  SCR -->|"deprecated"| PBC_SHIM["md_10mer_mmml_pbc_suite.py\n→ md_pbc_suite.ase"]
```

### Scripts → CLI migration

| Former `scripts/` | Current location |
|-------------------|------------------|
| `md_10mer_mmml_pbc_suite.py` | `mmml.cli.run.md_pbc_suite.ase` (shim in `scripts/`) |
| `md_10mer_mmml_pbc_suite_jaxmd.py` | `mmml.cli.run.md_pbc_suite.jaxmd` |
| `test_orbax_checkpoint_cluster.py` | `mmml.cli.run.md_pbc_suite.cluster` |
| `meoh_dimer_lambda_ti.py` | `mmml md-system --setup lambda_ti` / `lambda_dynamics` |
| `meoh_dimer_lambda_mbar.py` | `mmml lambda-mbar` |
| `pycharmm_two_residue_sample.py` | `mmml pycharmm-two-residue-sample` |

Research / dev scripts that may stay under `scripts/`: `scan_meoh_dimer_*`, `benchmark_mmml_scaling.py`, shell helpers, data README.

---

## Regenerating import statistics

```bash
cd /path/to/mmml
python3 <<'PY'
import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path("mmml")

def top_pkg(mod: str) -> str:
    parts = mod.split(".")
    return ".".join(parts[:2]) if len(parts) > 2 else mod

def file_to_mod(path: Path) -> str:
    rel = path.relative_to(ROOT.parent)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    return ".".join(parts)[:-3]

def parse_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append(a.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out

pkg_edges: dict[tuple[str, str], int] = defaultdict(int)
for path in ROOT.rglob("*.py"):
    if "viewer" in path.parts or "__pycache__" in path.parts:
        continue
    src = top_pkg(file_to_mod(path))
    if not src.startswith("mmml"):
        continue
    for imp in parse_imports(path):
        if imp.startswith("mmml"):
            tgt = top_pkg(imp)
            if src != tgt:
                pkg_edges[(src, tgt)] += 1

for (a, b), c in sorted(pkg_edges.items(), key=lambda x: -x[2]):
    print(f"{c:4d}  {a}  ->  {b}")
PY
```

---

## See also

- [Development](development.md) — tests, MkDocs workflow
- [Getting started](getting-started.md) — install and first runs
- [API reference](api.md) — generated API docs
