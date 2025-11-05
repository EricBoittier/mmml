# 1. Explore
python -m mmml.cli.explore_data raw.npz --detailed --plots

# 2. Clean
python -m mmml.cli.clean_data raw.npz -o clean.npz --no-check-distances

# 3. Split
python -m mmml.cli.split_dataset clean.npz -o splits/ --convert-units

# 4. Train (auto-detects num_atoms!)
python -m mmml.cli.make_training --data splits/data_train.npz --ckpt_dir ckpts/run1

# 5. Monitor
python -m mmml.cli.plot_training ckpts/run1/history.json --dpi 300

# 6. Inspect
python -m mmml.cli.inspect_checkpoint --checkpoint ckpts/run1

# 7. Test
python -m mmml.cli.calculator --checkpoint ckpts/run1 --test-molecule CO2

# 8. Evaluate
python -m mmml.cli.evaluate_model --checkpoint ckpts/run1 --data splits/data_test.npz

# 9. Dynamics
python -m mmml.cli.dynamics --checkpoint ckpts/run1 --molecule CO2 --optimize --frequencies

# 10. Visualize
python -m mmml.cli.convert_npz_traj clean.npz -o structures.traj