import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = all_data

# data = pd.read_csv('data.csv')
def _unpack(a, k):
    return np.array([_[k].get() if (not type(_[k]) == np.ndarray) and (not type(_[k]) is np.float64)  else _[k] for _ in a])

# Unpack all variables
variables = ['Ef', 'D', 'E', 'A', 'F', 'H', 'freq', 'intensity', 'dDdR', 'R', 'Z']
data_dict = {var: _unpack(data, var) for var in variables}

# Calculate additional physical quantities (using vectorized operations)
force_norm = np.linalg.norm(data_dict['F'], axis=-1)
dipole_norm = np.linalg.norm(data_dict['D'], axis=-1)
energy = data_dict['E']

# Separate variables by plot type
tensor_vars = ['A', 'D', 'F', 'dDdR']
scalar_vars = ['Ef', 'E', 'freq', 'intensity', 'Z']

# Define correlation pairs
correlation_pairs = [
    (energy, force_norm, 'Energy', 'Force Norm'),
    (energy, dipole_norm, 'Energy', 'Dipole Norm'),
    (force_norm, dipole_norm, 'Force Norm', 'Dipole Norm'),
    (data_dict['freq'], data_dict['intensity'], 'Frequency', 'Intensity'),
    (energy, data_dict['Z'], 'Energy', 'Nuclear Charge')
]

# Calculate layout
n_tensor_plots = sum(data_dict[var].ndim - 1 for var in tensor_vars)
n_scalar_plots = len(scalar_vars)
n_correlation_plots = len(correlation_pairs)
n_plots = n_tensor_plots + n_scalar_plots + n_correlation_plots
n_cols = min(4, n_plots)
n_rows = int(np.ceil(n_plots / n_cols))

# Create figure
fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
plot_idx = 1

# Plot tensors
for var in tensor_vars:
    data = data_dict[var]
    if data.ndim > 2:
        data = data.mean(axis=0)
    
    if var == 'H':
        n_atoms = int(np.sqrt(data.shape[-1] / 3))
        data = data.reshape(-1, n_atoms*3, n_atoms*3)
    
    ax = plt.subplot(n_rows, n_cols, plot_idx)
    im = ax.matshow(data, cmap='coolwarm')
    plt.colorbar(im, ax=ax)
    plt.title(f'{var} Matrix')
    ax.xaxis.set_ticks_position('bottom')
    plot_idx += 1

# Plot histograms (with optimized bin count)
for var in scalar_vars:
    data = data_dict[var].ravel()
    ax = plt.subplot(n_rows, n_cols, plot_idx)
    n_bins = int(np.log2(len(data)) + 1)
    plt.hist(data, bins=n_bins, alpha=0.7)
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title(f'{var} Distribution')
    plot_idx += 1

# Plot correlations (with downsampling for large datasets)
for x, y, xlabel, ylabel in correlation_pairs:
    ax = plt.subplot(n_rows, n_cols, plot_idx)
    
    # Downsample if more than 10000 points
    if len(x) > 10000:
        idx = np.random.choice(len(x), 10000, replace=False)
        x_plot = x[idx]
        y_plot = y[idx]
    else:
        x_plot = x
        y_plot = y
    
    plt.scatter(x_plot, y_plot, alpha=0.5, s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Calculate correlation coefficient using full dataset
    corr = np.corrcoef(x.ravel(), y.ravel())[0,1]
    plt.title(f'{xlabel} vs {ylabel}\nr = {corr:.2f}')
    
    # Add trend line using full dataset
    z = np.polyfit(x.ravel(), y.ravel(), 1)
    p = np.poly1d(z)
    x_range = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plot_idx += 1

plt.tight_layout()
plt.show()

# Additional plot for R matrix coordinates
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.scatter(data_dict['R'][:, 0], data_dict['R'][:, 1], data_dict['R'][:, 2], c=data_dict['Z'], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('R Matrix Coordinates (colored by Z values)')
plt.show()