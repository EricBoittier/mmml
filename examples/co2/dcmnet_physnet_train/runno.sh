<<<<<<< HEAD
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=0 && uv run python compare_models.py     --train-efd energies_forces_dipoles_train.npz --train-esp grids_esp_train.npz     --valid-efd energies_forces_dipoles_valid.npz --valid-esp grids_esp_valid.npz --epochs 1000 --batch-size 100 --comparison-name test1 
=======

export CUDA_VISIBLE_DEVICES=0 && uv run python compare_models.py     --train-efd energies_forces_dipoles_train.npz --train-esp grids_esp_train.npz     --valid-efd energies_forces_dipoles_valid.npz --valid-esp grids_esp_valid.npz --epochs 1000 --batch-size 100 --comparison-name test1

>>>>>>> 573c5ecf (dsafg)
=======
echo $seed && export CUDA_VISIBLE_DEVICES=0 && uv run python compare_models.py     --train-efd energies_forces_dipoles_train.npz --train-esp grids_esp_train.npz     --valid-efd energies_forces_dipoles_valid.npz --valid-esp grids_esp_valid.npz --epochs 1000 --batch-size 100 --comparison-name test$seed --seed $seed 
>>>>>>> fd2b6c92 (asdf)
