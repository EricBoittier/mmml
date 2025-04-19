bash ~/mmml/setup/install.sh
uv run python ~/mmml/mmml/pyscf4gpuInterface/calcs.py --output qm/initial.npz --mol xyz/initial.xyz --optimize --xc PBE0 --basis cc-pVTZ --hessian --thermo --gradient --harmonic > qm/initial.npz
            