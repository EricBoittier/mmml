#!/bin/bash
# Example commands for running DCMNet training with Hydra

set -e

echo "DCMNet Training Examples"
echo "========================"
echo ""

# Example 1: Quick test run
echo "1. Quick test run (fast, for debugging)"
echo "   Command: python train.py experiment=quick_test"
echo ""

# Example 2: Standard training
echo "2. Standard training with default config"
echo "   Command: python train.py"
echo ""

# Example 3: Bootstrap training
echo "3. Bootstrap training with 10 iterations"
echo "   Command: python train.py training=bootstrap training.n_bootstrap=10"
echo ""

# Example 4: Large model
echo "4. Train large model"
echo "   Command: python train.py model=large"
echo ""

# Example 5: Hyperparameter sweep
echo "5. Hyperparameter sweep (grid search)"
echo "   Command: python train.py -m model.features=64,128 training.learning_rate=1e-4,5e-4"
echo ""

# Example 6: Optuna sweep (requires hydra-optuna-sweeper)
echo "6. Smart hyperparameter search with Optuna"
echo "   Command: python train.py -m --config-name=sweep_optuna"
echo ""

# Example 7: Evaluation
echo "7. Evaluate trained model"
echo "   Command: python evaluate.py checkpoint_path=outputs/YYYY-MM-DD/HH-MM-SS/final_params.npz"
echo ""

# Prompt user to select an example
echo "Enter example number to run (1-7), or 'q' to quit:"
read -r choice

case $choice in
    1)
        echo "Running quick test..."
        python train.py experiment=quick_test
        ;;
    2)
        echo "Running standard training..."
        python train.py
        ;;
    3)
        echo "Running bootstrap training..."
        python train.py training=bootstrap training.n_bootstrap=10
        ;;
    4)
        echo "Running large model training..."
        python train.py model=large
        ;;
    5)
        echo "Running hyperparameter sweep..."
        python train.py -m model.features=64,128 training.learning_rate=1e-4,5e-4 seed=42,43
        ;;
    6)
        echo "Running Optuna sweep..."
        python train.py -m --config-name=sweep_optuna
        ;;
    7)
        echo "Please provide the checkpoint path:"
        read -r checkpoint_path
        python evaluate.py checkpoint_path="$checkpoint_path"
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

