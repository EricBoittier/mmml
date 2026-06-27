setup: pbc_npt
backend: pycharmm
composition: DCM:32
output_dir: REPLACE_OUT
box_size: 25.0
spacing: 4.0
temperature: 300.0
pressure: 1.0
checkpoint: REPLACE_CKPT

md_stages: mini
mini_nstep: 40
skip_jit_warmup: false
ml_batch_size: 128
ml_gpu_count: 1

charmm_pre_minimize: true
charmm_sd_steps: 50
charmm_abnr_steps: 50
bonded_mm_mini: true
bonded_mm_mini_steps: 50
max_grms_before_dyn: 100.0
no_echeck: true
mlpot_profile: true
quiet: true

mm_switch_on: 8.0
mm_switch_width: 5.0
ml_switch_width: 1.5
include_mm: true
