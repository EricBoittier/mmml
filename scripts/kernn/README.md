# KerNN (legacy location)

The PyTorch KerNN prototype that lived here has been removed.

Use the JAX/Flax package instead:

- Package: [`mmml.models.kernnn`](../../mmml/models/kernnn/)
- Train / eval: `mmml kernnn-train`, `mmml kernnn-evaluate`
- ASE / scans: `--calculator kernnn --checkpoint …`
- NEB: `mmml neb --calculator kernnn --checkpoint …`
- Umbrella: `mmml umbrella-sample --model kernnn --checkpoint …`
- DMC: `mmml dmc --model kernnn --natm 4 --checkpoint …`
- Hybrid MLpot / md-system: pass a KerNN JSON checkpoint (`model_type: kernnn`); auto-detected by `setup_calculator`

See [`mmml/models/kernnn/README.md`](../../mmml/models/kernnn/README.md) for the full API.
