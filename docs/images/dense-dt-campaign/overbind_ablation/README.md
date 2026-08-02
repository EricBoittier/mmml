# DCM–DCM overbinding ablations

Epoch222 hybrid checkpoint. Ablation grid: 64 rays × 36 COM distances
(`scripts/slurm/dense_dt_campaign/ablate_overbind.py`).

Literature DCM dimer wells are about **−3 to −5 kcal/mol**.

**Contact policy:** COM–COM `r` alone is not steric for DCM. Ablation tables
below still quote unfiltered ray minima (includes Cl/H clashes at short `r`).
Clash-filtered soft-well median for the epoch222 baseline is about **−3.7
kcal/mol** (`dmin ≥ 2 Å`; see `docs/images/dense-dt-campaign/dimer_scans/` and
`DEFAULT_ORIENT_MIN_CONTACT_A`). Re-run `ablate_overbind.py` to refresh the
grid with the same cut.

## Results (soft well = per-ray min at r ≥ 3.4 Å; unfiltered)

| Run | Soft well median | Contact ray-min median | ML full below | Role |
|---|---:|---:|---:|---|
| baseline train handoff (on=8) | −8.9 | −30.8 | 6.5 Å | overbinds (raw) |
| `es_off_on8` | −8.9 | −30.8 | 6.5 Å | ES not the driver |
| **`handoff_on5_w1p5` (lever 2 soft)** | **−4.4** | −30.8 | 3.5 Å | **campaign default** |
| `handoff_on4p5_w1` | −3.0 | −30.8 | 3.5 Å | still contact-deep |
| `contact_on3p5_w1p5` | −1.1 | **−1.1** | 2.0 Å | kills contacts, underbinds |

## Lever 2 — earlier MM handoff (shipped in campaign)

`run_one.sh` now passes `--mm-switch-on 5 --ml-switch-width 1.5` (`DDC_HANDOFF=soft`).
This is a **deploy mismatch** vs the epoch222 train taper (8/1.5/5); the calculator
warns on purpose. Soft wells move into the literature band and should reduce the
dense NVT droplet drive from medium-range overbinding.

## How to fix contact rays (r ≲ 3.4 Å, −30 kcal)

Contact minima sit where `ml_scale=1` even under soft lever-2 (ML full below 3.5 Å).

| Approach | Soft well | Contact wells | Status |
|---|---|---|---|
| Deploy `DDC_HANDOFF=contact` (on=3.5) | underbinds (~−1) | fixed | diagnostic only |
| **Retrain at on=5** with dimer soft targets | keep ~−4 | should unlearn | **recommended** |
| Stronger short-range wall (`r_on`≫1 Å) | may distort | partial | not preferred |

Recommended next train:

```bash
uv run mmml physnet-train \
  --config examples/hybrid_mm_charges/train_fixed_lj_scales.yaml \
  --data artifacts/lj_scales/dataset_cgenff.npz \
  --physnet-checkpoint artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json \
  --hybrid-mm --learn-mm-lj-scales --lr-solver mic \
  --mm-switch-on 5.0 --ml-switch-width 1.5 --mm-switch-width 5.0 \
  --tag hybrid_mm_lever2_on5_ft --num-epochs 50
```

Then redeploy MD with the **same** handoff (parity, no mismatch warning).

## Figures

- `overbind_ablation_compare.png` — mean E_int curves + soft-well bars
- `overbind_handoff_components.png` — ML vs MM for early-handoff variants
