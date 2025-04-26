import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# data = pd.read_csv('data.csv')
def _unpack(a, k):
    return [_[k] for _ in a]

freq = _unpack(data, 'freq')
intensity = _unpack(data, 'intensity')
A = _unpack(data, 'A')
F = _unpack(data, 'F')
R = _unpack(data, 'R')
Z = _unpack(data, 'Z')


# Create figure with multiple subplots
plt.style.use('seaborn')
fig = plt.figure(figsize=(15, 10))

# 1. Plot frequencies vs intensities
ax1 = plt.subplot(221)
plt.scatter(freq, intensity, alpha=0.6)
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.title('Frequency vs Intensity')

# 2. Plot eigenvalues (diagonal of A matrix)
ax2 = plt.subplot(222)
eigenvals = np.diag(A)
plt.bar(range(len(eigenvals)), eigenvals)
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues Distribution')

# 3. Create heatmap of A matrix
ax3 = plt.subplot(223)
sns.heatmap(A, annot=True, cmap='coolwarm', center=0)
plt.title('A Matrix Heatmap')

# 4. Create heatmap of F matrix
ax4 = plt.subplot(224)
sns.heatmap(F, annot=True, cmap='coolwarm', center=0)
plt.title('F Matrix Heatmap')

plt.tight_layout()
plt.show()

# Additional plot for R matrix coordinates
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.scatter(R[:, 0], R[:, 1], R[:, 2], c=Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('R Matrix Coordinates (colored by Z values)')
plt.show()