import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_3d_scatter_views(coords, color_values, cmap_name='bwr', angles=[(20, 45), (90, -90), (0, 0)], vmin=None, vmax=None):
    """
    Plots a 3D scatter plot from coords and color_values, shown from 3 different angles.
    
    Parameters:
    - coords: (n, 3) array of xyz coordinates
    - color_values: (n,) array of scalar values used for colormap
    - cmap_name: string, name of matplotlib colormap
    - angles: list of (elev, azim) tuples for different 3D views
    """
    assert coords.shape[1] == 3, "Input coords must be of shape (n, 3)"
    assert coords.shape[0] == len(color_values), "color_values must have same length as number of points"
    if vmin is None:
        vmin = color_values.min()
    if vmax is None:
        vmax=color_values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    mapped_colors = cmap(norm(color_values))

    fig = plt.figure(figsize=(18, 6))
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=mapped_colors, s=50, edgecolor='k', alpha=0.9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {i+1} (elev={elev}, azim={azim})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    

    # plot the distribution of esp values
    new_ax1 = fig.add_subplot(2, 4, 4)
    ax = plot_esp_distribution(color_values, cmap_name=cmap_name, vmin=vmin, vmax=vmax, ax=new_ax1)
    new_ax1.set_title("Distribution of ESP values")

    # Add shared colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(color_values)
    cbar = plt.colorbar(mappable, ax=fig.axes, orientation='horizontal', fraction=0.02, pad=0.4)
    cbar.set_label("ESP [(Hartree/Bohr) / $e$]")


    plt.tight_layout()
    plt.show()


def plot_esp_distribution(esp_values, cmap_name='bwr', vmin=None, vmax=None, ax=None):
    """
    Plots the distribution of ESP values using a histogram.
    
    Parameters:
    - esp_values: (n,) array of ESP values
    - cmap_name: string, name of matplotlib colormap
    - vmin: float, minimum value for color mapping
    - vmax: float, maximum value for color mapping
    - ax: matplotlib axis object, optional
    """
    if ax is None:
        ax = plt.gca()
    if vmin is None:
        vmin = esp_values.min()
    if vmax is None:
        vmax=esp_values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    mapped_colors = cmap(norm(esp_values))
    ax.hist(esp_values, bins=50, edgecolor='black')
    # set the color of the histogram bars to the mapped colors
    for i, bar in enumerate(ax.patches):
        bar.set_color(mapped_colors[i*len(esp_values)//50])
    # add mean, median, and std of esp values
    mean = np.mean(esp_values)
    median = np.median(esp_values)
    std = np.std(esp_values)

    ax.axvline(mean, color='red', linestyle='--', label=f'Mean {mean:.2f}')
    ax.axvline(median, color='green', linestyle='--', label=f'Median {median:.2f}')
    ax.axvline(std, color='blue', linestyle='--', label=f'Std {std:.2f}')
    ax.legend()
    ax.set_xlabel("ESP [(Hartree/Bohr) / $e$]")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of ESP values")




    return ax

