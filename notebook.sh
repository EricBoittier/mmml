module purge && source .venv/bin/activate && module load cudnn; module load gcc; module load charmm && uv run --with jupyter jupyter lab --no-browser --port 9856
