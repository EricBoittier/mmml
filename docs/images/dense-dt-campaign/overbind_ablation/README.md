# DCM–DCM overbinding ablations

Epoch222 hybrid checkpoint. Ablation grid: 64 rays × 36 COM distances
(`scripts/slurm/dense_dt_campaign/ablate_overbind.py`).

Literature DCM dimer wells are about **−3 to −5 kcal/mol**.

## Results (soft well = per-ray min at r ≥ 3.4 Å)

| Run | Soft well mean | Soft well median | Contact ray-min mean | ML full below |
|---|---:|---:|---:|---:|
| baseline (96-ray campaign) | −10.2 | −9.9 | −30.4 | 6.5 Å |
| `es_off_on8` (lever 1) | −8.4 | −8.9 | −30.1 | 6.5 Å |
| `handoff_on6_w1p5` | −8.4 | −8.9 | −30.1 | 4.5 Å |
| `handoff_on5_w1p5` (lever 2) | −5.4 | −4.4 | −29.4 | 3.5 Å |
| `handoff_on4p5_w1` (lever 2) | −4.6 | −3.0 | −29.4 | 3.5 Å |
| `ft_early_handoff_*` | ≳ −1 | ≳ −1 | ≳ −1 | — |

## Takeaways

1. **Lever 1 (ES-off):** PhysNet Coulomb off does **not** change the wells.
   Overbinding is neural local ML energy. A 5-epoch CPU warm-start fine-tune
   with earlier handoff collapsed the interaction (underbinding) — needs a
   proper GPU retrain, not a smoke FT.
2. **Lever 2 (earlier MM handoff):** Deploy-time `mm_switch_on=5` / `4.5`
   moves **soft** wells into the literature band, but **~94% of rays still
   minimize at r ≲ 3.4 Å** where ML remains fully on (−30 kcal contact wells).
   Turning MM on near 3.5–4 Å also produces rare LJ explosions (mean curves
   are outlier-sensitive; medians are the right summary).

## Figures

- `overbind_ablation_compare.png` — mean E_int curves + soft-well bars
- `overbind_handoff_components.png` — ML vs MM for early-handoff variants
EOF