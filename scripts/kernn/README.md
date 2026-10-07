# KerNN (legacy location)

The PyTorch KerNN prototype that lived here has been removed.

Use the JAX/Flax package instead:

- Package: [`karml.models.kernnn`](../../karml/models/kernnn/)
- Train / eval: `karml kernnn-train`, `karml kernnn-evaluate`
- ASE / scans: `--calculator kernnn --checkpoint …`
- NEB: `karml neb --calculator kernnn --checkpoint …`
- Umbrella: `karml umbrella-sample --model kernnn --checkpoint …`
- DMC: `karml dmc --model kernnn --natm 4 --checkpoint …`
- Hybrid MLpot / md-system: pass a KerNN JSON checkpoint (`model_type: kernnn`); auto-detected by `setup_calculator`

See [`karml/models/kernnn/README.md`](../../karml/models/kernnn/README.md) for the full API.
