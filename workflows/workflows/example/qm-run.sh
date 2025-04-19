bash ~/mmml/setup/install.sh
uv run python ~/mmml/mmml/pyscf4gpuInterface/calcs.py --output qm/run.npz --mol xyz/run.xyz --optimize --xc PBE0 --basis cc-pVTZ --hessian --thermo --gradient --harmonic > qm/run.npz
            