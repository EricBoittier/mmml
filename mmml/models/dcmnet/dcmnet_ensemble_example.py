# ================================================
# DCMNET MCTS Integration Example
# ================================================
"""
Complete example showing how to integrate MCTS-based model selection
with your existing DCMNET workflow.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, List
import pandas as pd

# Import your existing DCMNET components
from dcmnet.models import DCM1, DCM2, DCM3, DCM4, dcm1_params, dcm2_params, dcm3_params, dcm4_params
from dcmnet.loss import esp_mono_loss
from dcmnet.electrostatics import calc_esp
from dcmnet_mcts import DCMNETSelectionEnv, DCMNET_MCTS, optimize_dcmnet_combination

class DCMNETEnsembleOptimizer:
    """
    High-level interface for optimizing DCMNET model ensembles using MCTS.
    """
    
    def __init__(self, 
                 molecular_data: Dict[str, Any],
                 esp_target: jnp.ndarray,
                 vdw_surface: jnp.ndarray,
                 available_models: List[Tuple[Any, Any]] = None):
        """
        Initialize the ensemble optimizer.
        
        Args:
            molecular_data: Molecular data dictionary
            esp_target: Target ESP values
            vdw_surface: VdW surface points
            available_models: List of (model, params) tuples
        """
        self.molecular_data = molecular_data
        self.esp_target = esp_target
        self.vdw_surface = vdw_surface
        
        # Default models (you can add DCM5-DCM7 when available)
        if available_models is None:
            self.available_models = [
                (DCM1, dcm1_params),
                (DCM2, dcm2_params),
                (DCM3, dcm3_params),
                (DCM4, dcm4_params),
            ]
        else:
            self.available_models = available_models
        
        self.best_combination = None
        self.best_loss = float('inf')
        self.optimization_history = []
    
    def evaluate_combination(self, model_selection: np.ndarray) -> float:
        """
        Evaluate a specific combination of models.
        
        Args:
            model_selection: Binary array indicating which models to use
            
        Returns:
            ESP loss for this combination
        """
        if np.sum(model_selection) == 0:
            return float('inf')
        
        # Combine predictions from selected models
        n_atoms = len(self.molecular_data['atomic_numbers'])
        combined_mono = jnp.zeros(n_atoms)
        combined_dipo = jnp.zeros((n_atoms, 3))
        
        n_selected = 0
        for i, (model, params) in enumerate(self.available_models):
            if i < len(model_selection) and model_selection[i]:
                try:
                    mono_pred, dipo_pred = model.apply(
                        params,
                        self.molecular_data['atomic_numbers'],
                        self.molecular_data['positions'],
                        self.molecular_data['dst_idx'],
                        self.molecular_data['src_idx']
                    )
                    combined_mono += mono_pred
                    combined_dipo += dipo_pred
                    n_selected += 1
                except Exception as e:
                    print(f"Warning: Could not evaluate model {i}: {e}")
        
        if n_selected == 0:
            return float('inf')
        
        # Average the predictions
        combined_mono /= n_selected
        combined_dipo /= n_selected
        
        # Calculate ESP loss
        try:
            esp_pred = calc_esp(
                combined_mono,
                combined_dipo,
                self.molecular_data['positions'],
                self.vdw_surface
            )
            loss = jnp.mean((esp_pred - self.esp_target) ** 2)
            return float(loss)
        except Exception as e:
            print(f"Error calculating ESP loss: {e}")
            return float('inf')
    
    def optimize_with_mcts(self, 
                          n_simulations: int = 1000,
                          temperature: float = 1.0,
                          n_restarts: int = 5) -> Tuple[np.ndarray, float]:
        """
        Optimize model combination using MCTS with multiple restarts.
        
        Args:
            n_simulations: Number of MCTS simulations per restart
            temperature: Temperature for action selection
            n_restarts: Number of independent optimization runs
            
        Returns:
            Tuple of (best_model_selection, best_esp_loss)
        """
        best_overall_selection = None
        best_overall_loss = float('inf')
        
        for restart in range(n_restarts):
            print(f"Starting MCTS optimization run {restart + 1}/{n_restarts}")
            
            # Create environment for this run
            env = DCMNETSelectionEnv(
                self.molecular_data,
                self.esp_target,
                self.vdw_surface,
                max_models=len(self.available_models)
            )
            
            # Simple policy-value function
            def policy_value_fn(env):
                legal_actions = env.legal_actions()
                
                # Heuristic policy: prefer simpler models (lower indices)
                policy_logits = np.array([-i * 0.1 for i in range(len(self.available_models))])
                legal_logits = policy_logits[legal_actions]
                exp_logits = np.exp(legal_logits - np.max(legal_logits))
                probs = exp_logits / np.sum(exp_logits)
                
                priors = {int(a): float(p) for a, p in zip(legal_actions, probs)}
                
                # Value: negative ESP loss
                esp_loss = env.get_esp_loss()
                value = -esp_loss if esp_loss != float('inf') else -1e6
                
                return priors, value
            
            # Initialize MCTS
            mcts = DCMNET_MCTS(
                policy_value_fn=policy_value_fn,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                root_noise_frac=0.25
            )
            
            # Perform search
            best_action = mcts.search(env, n_simulations=n_simulations, temperature=temperature)
            
            # Get final result
            final_env = env.step(best_action)
            final_loss = final_env.get_esp_loss()
            final_selection = final_env.selected_models
            
            print(f"Run {restart + 1}: Loss = {final_loss:.6f}, Selection = {final_selection}")
            
            # Track best overall result
            if final_loss < best_overall_loss:
                best_overall_loss = final_loss
                best_overall_selection = final_selection.copy()
            
            # Store in history
            self.optimization_history.append({
                'run': restart + 1,
                'loss': final_loss,
                'selection': final_selection.copy(),
                'selected_models': [f"DCM{i+1}" for i in range(len(self.available_models)) if final_selection[i]]
            })
        
        self.best_combination = best_overall_selection
        self.best_loss = best_overall_loss
        
        return best_overall_selection, best_overall_loss
    
    def compare_all_combinations(self) -> pd.DataFrame:
        """
        Evaluate all possible combinations of models (for small numbers of models).
        
        Returns:
            DataFrame with results for all combinations
        """
        n_models = len(self.available_models)
        results = []
        
        # Generate all possible combinations
        for i in range(1, 2**n_models):  # Skip empty combination
            # Convert to binary representation
            selection = np.array([(i >> j) & 1 for j in range(n_models)])
            
            # Evaluate this combination
            loss = self.evaluate_combination(selection)
            
            # Get model names
            selected_models = [f"DCM{j+1}" for j in range(n_models) if selection[j]]
            
            results.append({
                'combination_id': i,
                'selection': selection,
                'selected_models': selected_models,
                'n_models': np.sum(selection),
                'esp_loss': loss
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('esp_loss')
        
        return df
    
    def get_best_models(self) -> List[str]:
        """Get list of best model names."""
        if self.best_combination is None:
            return []
        
        return [f"DCM{i+1}" for i in range(len(self.available_models)) if self.best_combination[i]]
    
    def print_results(self):
        """Print optimization results."""
        if self.best_combination is None:
            print("No optimization results available.")
            return
        
        print(f"\n=== DCMNET Ensemble Optimization Results ===")
        print(f"Best ESP Loss: {self.best_loss:.6f}")
        print(f"Best Model Combination: {self.get_best_models()}")
        print(f"Number of Models Used: {np.sum(self.best_combination)}")
        
        print(f"\n=== Optimization History ===")
        for result in self.optimization_history:
            print(f"Run {result['run']}: Loss = {result['loss']:.6f}, "
                  f"Models = {result['selected_models']}")

# =========================================================
# Example Usage with Real Data
# =========================================================

def load_example_data():
    """Load example molecular data for testing."""
    # This is a placeholder - replace with your actual data loading
    molecular_data = {
        'atomic_numbers': jnp.array([6, 1, 1, 1, 1]),  # CH4 example
        'positions': jnp.array([[0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [-1.0, -1.0, 1.0],
                               [-1.0, 1.0, -1.0],
                               [1.0, -1.0, -1.0]]),
        'dst_idx': jnp.array([0, 0, 0, 0]),
        'src_idx': jnp.array([1, 2, 3, 4])
    }
    
    # Example ESP target and VdW surface
    esp_target = jnp.random.normal(0, 1, (100,))
    vdw_surface = jnp.random.normal(0, 2, (100, 3))
    
    return molecular_data, esp_target, vdw_surface

def main():
    """Main example function."""
    print("Loading example data...")
    molecular_data, esp_target, vdw_surface = load_example_data()
    
    print("Initializing DCMNET ensemble optimizer...")
    optimizer = DCMNETEnsembleOptimizer(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface
    )
    
    print("Running MCTS optimization...")
    best_selection, best_loss = optimizer.optimize_with_mcts(
        n_simulations=500,
        temperature=1.0,
        n_restarts=3
    )
    
    print("Printing results...")
    optimizer.print_results()
    
    # Compare with exhaustive search (for small number of models)
    print("\nComparing with exhaustive search...")
    if len(optimizer.available_models) <= 4:  # Only for small numbers
        comparison_df = optimizer.compare_all_combinations()
        print("Top 5 combinations:")
        print(comparison_df.head())
    else:
        print("Too many models for exhaustive search. MCTS is the way to go!")

if __name__ == "__main__":
    main()
