"""
Example: Training DCMNet with Multi-Batch Support

This example demonstrates how to use the new multi-batch training system.
"""

import jax
from pathlib import Path

# Simple one-line training
from mmml.dcmnet.dcmnet.train_runner import run_training

def example_basic_training():
    """Basic training example with gradient accumulation."""
    print("\n" + "="*80)
    print("Example 1: Basic Multi-Batch Training")
    print("="*80 + "\n")
    
    params, loss, exp_dir = run_training(
        name="basic_example",
        num_epochs=50,
        batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        learning_rate=1e-4,
        num_train=500,
        num_valid=100
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {loss:.6e}")
    print(f"Results saved to: {exp_dir}")
    
    return params, exp_dir


def example_advanced_training():
    """Advanced training with LR schedule and custom config."""
    print("\n" + "="*80)
    print("Example 2: Advanced Training with LR Schedule")
    print("="*80 + "\n")
    
    params, loss, exp_dir = run_training(
        name="advanced_example",
        num_epochs=100,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        use_lr_schedule=True,
        lr_schedule_type="cosine",
        warmup_epochs=10,
        esp_w=1000.0,
        chg_w=1.0,
        n_dcm=3,                        # More multipoles
        features=32,                    # Larger model
        use_grad_clip=True,
        grad_clip_norm=1.0,
        save_every_n_epochs=10,
        num_train=2000,
        num_valid=400
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {loss:.6e}")
    print(f"Results saved to: {exp_dir}")
    
    return params, exp_dir


def example_analysis():
    """Analyze trained model."""
    print("\n" + "="*80)
    print("Example 3: Model Analysis")
    print("="*80 + "\n")
    
    from mmml.dcmnet.dcmnet.analysis_multibatch import (
        analyze_checkpoint, 
        print_analysis_summary,
        batch_analysis_summary
    )
    from mmml.dcmnet.dcmnet.analysis import create_model
    from mmml.dcmnet.dcmnet.data import prepare_datasets
    
    # Load test data
    key = jax.random.PRNGKey(42)
    _, test_data = prepare_datasets(
        key,
        num_train=100,
        num_valid=100,
        filename=["esp2000.npz"]
    )
    
    # Create model
    model = create_model(n_dcm=2, features=16)
    
    # Analyze the basic example
    exp_dir = Path("experiments/basic_example")
    if exp_dir.exists():
        results = batch_analysis_summary(
            exp_dir=exp_dir,
            model=model,
            test_data=test_data
        )
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {exp_dir}/analysis")
    else:
        print(f"Experiment directory not found: {exp_dir}")
        print("Run example_basic_training() first!")


def example_resume_training():
    """Resume training from checkpoint."""
    print("\n" + "="*80)
    print("Example 4: Resume Training from Checkpoint")
    print("="*80 + "\n")
    
    checkpoint_path = Path("experiments/basic_example/checkpoints/checkpoint_latest.pkl")
    
    if checkpoint_path.exists():
        params, loss, exp_dir = run_training(
            name="basic_example",          # Same experiment name
            num_epochs=100,                # Train for more epochs
            restart_checkpoint=checkpoint_path,
            # Other parameters will be loaded from checkpoint config
        )
        
        print(f"\nResumed training complete!")
        print(f"Best validation loss: {loss:.6e}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run example_basic_training() first!")


def example_compare_experiments():
    """Compare multiple trained models."""
    print("\n" + "="*80)
    print("Example 5: Compare Multiple Experiments")
    print("="*80 + "\n")
    
    from mmml.dcmnet.dcmnet.analysis_multibatch import compare_checkpoints
    from mmml.dcmnet.dcmnet.analysis import create_model
    from mmml.dcmnet.dcmnet.data import prepare_datasets
    
    # Load test data
    key = jax.random.PRNGKey(42)
    _, test_data = prepare_datasets(
        key,
        num_train=100,
        num_valid=100,
        filename=["esp2000.npz"]
    )
    
    # Create model
    model = create_model(n_dcm=2, features=16)
    
    # Compare checkpoints
    checkpoints = [
        Path("experiments/basic_example/checkpoints/checkpoint_best.pkl"),
        Path("experiments/advanced_example/checkpoints/checkpoint_best.pkl"),
    ]
    
    # Check if checkpoints exist
    existing_checkpoints = [c for c in checkpoints if c.exists()]
    
    if len(existing_checkpoints) >= 2:
        df = compare_checkpoints(
            checkpoint_paths=existing_checkpoints,
            model=model,
            test_data=test_data,
            labels=["Basic", "Advanced"]
        )
        
        print("\nComparison Results:")
        print(df.to_string())
    else:
        print("Need at least 2 trained checkpoints to compare!")
        print("Run example_basic_training() and example_advanced_training() first!")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("DCMNet Multi-Batch Training Examples")
    print("="*80 + "\n")
    
    # Example 1: Basic training
    print("Running Example 1...")
    params1, exp_dir1 = example_basic_training()
    
    # Example 3: Analysis
    print("\nRunning Example 3...")
    example_analysis()
    
    # Example 4: Resume training (optional)
    # print("\nRunning Example 4...")
    # example_resume_training()
    
    # Example 2: Advanced training (takes longer)
    # print("\nRunning Example 2...")
    # params2, exp_dir2 = example_advanced_training()
    
    # Example 5: Compare experiments
    # print("\nRunning Example 5...")
    # example_compare_experiments()
    
    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run individual examples
    # example_basic_training()
    # example_advanced_training()
    # example_analysis()
    # example_resume_training()
    # example_compare_experiments()
    
    # Or run all examples
    main()

