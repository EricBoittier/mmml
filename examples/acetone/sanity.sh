module purge

source ~/.bashrc

module load gcc

# 2) activate the project venv (this must set $VIRTUAL_ENV)
source ~/mmml/.venv/bin/activate

UUID="test-70821ae3-d06a-4c87-9a2b-f5889c298376"

# jax cuda12
CUDA_VISIBLE_DEVICES=1 python "demo.py" \
	  --dataset "/pchem-data/meuwly/boittier/home/mmml/mmml/data/fixed-acetone-only_MP2_21000.npz" \
	    --checkpoint "/pchem-data/meuwly/boittier/home/mmml/mmml/physnetjax/ckpts/$UUID" \
	      --units eV \
	        --sample-index 0 \
		  --n-monomers 20 \
		    --atoms-per-monomer 10 \
		      --ml-cutoff 2.5 \
		        --mm-switch-on 5.0 \
			  --mm-cutoff 7.0 \
			    --include-mm \
			      --output "acetone/results.json" \
			        --pdbfile "init-packmol.pdb"

