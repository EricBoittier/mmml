source settings.source

python -m mmml.cli.opt_mmml \
   --dataset $DATA \
   --pdbfile "pdb/init-packmol.pdb" \
   --checkpoint $CHECKPOINT \
   --n-monomers 2 \
   --n-atoms-monomer 10 \
   --ml-cutoff-grid 1.0 \
   --mm-switch-on-grid 5.0 \
   --mm-cutoff-grid 1,2,3,4 \
   --energy-weight 1.0 \
   --force-weight 1.0 \
   --max-frames 100 \
   --include-mm \
   --out cutoff_opt.json
