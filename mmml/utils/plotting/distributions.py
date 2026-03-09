import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

def pretty_histogram(data, bins=None, alpha=0.7, edgecolor='black', title=None):
    """
    Create a pretty histogram of the data.
    """

    # if bins is None, use the best bin size for the data
    if bins is None:
        bins = np.histogram_bin_edges(data, bins='auto')

    plt.hist(data, bins=bins, alpha=alpha, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    if title is not None:   
        plt.title(title)

    plt.show()