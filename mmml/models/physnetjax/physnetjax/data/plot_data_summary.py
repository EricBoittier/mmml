import numpy as np
import matplotlib.pyplot as plt

from physnetjax.utils.enums import (
    check_keys,
    KEY_TRANSLATION,
    Z_KEYS,
    R_KEYS,
    F_KEYS,
    N_KEYS,
    D_KEYS,
    E_KEYS,
    COM_KEYS,
    ESP_GRID_KEYS,
    ESP_KEYS,
    Q_KEYS,
)


def plot_data_summary(data: dict):
    """
    Plot a summary of the data.
    """
    keys = list(data.keys())
    print(keys)

    for key in keys:
        if check_keys(R_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Positions')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(N_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Number of atoms')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(Z_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Atomic numbers')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(F_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Forces')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(E_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Energy')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(COM_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Center of mass')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(ESP_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('ESP')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(ESP_GRID_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('ESP grid')
            plt.ylabel('Frequency')
            plt.show()

        elif check_keys(Q_KEYS, key):
            plt.figure(figsize=(10, 5))
            plt.hist(data[key].flatten(), bins=100)
            plt.xlabel('Quadrupole')
            plt.ylabel('Frequency')
            plt.show()

