#!/usr/bin/env python3
"""
Quick example: Train charge predictor on CO2 data

This script demonstrates how to train the gradient boosting charge predictor
using the CO2 charge data.
"""

from pathlib import Path
from train_charge_predictor import load_charge_data, train_charge_predictor

# Path to your data
data_file = Path(__file__).parent.parent / "detailed_charges" / "df_charges_long.csv"

print("="*70)
print("CO2 Charge Predictor Training Example")
print("="*70)

# Load data - you can choose different schemes: Hirshfeld, VDD, Becke, etc.
# and levels: hf, mp2
print("\nLoading data...")
R, Z, mono = load_charge_data(data_file, scheme='Hirshfeld', level='hf')

# Train models
print("\nTraining models...")
results = train_charge_predictor(
    R=R,
    Z=Z,
    mono=mono,
    test_size=0.2,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    save_path=Path(__file__).parent / "charge_predictor_hirshfeld.pkl"
)

print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print(f"\nModel saved to: charge_predictor_hirshfeld.pkl")
print("\nTo use with DCMNet training:")
print("  from train_charge_predictor_usage import create_mono_imputation_fn_from_gb")
print("  mono_imputation_fn = create_mono_imputation_fn_from_gb('charge_predictor_hirshfeld.pkl')")
print("  train_model(..., mono_imputation_fn=mono_imputation_fn)")

