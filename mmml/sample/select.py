import numpy as np
from scipy.spatial.distance import pdist, squareform

def select_most_unique_samples(data, k):
    """
    Select k most unique samples based on pairwise distances using a greedy approach.
    
    Parameters:
    - data: ndarray of shape (n_samples, n_features)
    - k: int, number of unique samples to select
    
    Returns:
    - selected_indices: List of selected sample indices
    """
    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(data))
    
    # Find the most isolated point (max avg distance to others)
    avg_distances = np.mean(dist_matrix, axis=1)
    first_index = np.argmax(avg_distances)
    
    selected_indices = [first_index]
    
    for _ in range(k - 1):
        # Compute minimum distance to selected points for each unselected sample
        min_distances = np.min(dist_matrix[selected_indices], axis=0)
        
        # Select the sample with the maximum of these minimum distances
        next_index = np.argmax(min_distances)
        selected_indices.append(next_index)
    
    return selected_indices