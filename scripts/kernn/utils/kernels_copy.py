# torch imports
import torch
import torch.nn as nn




def get_1D_kernels_k33(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker33 = scale*(3.0 / (20.0 * xl ** 4) - 6.0 / 35.0 * xs / xl ** 5 + 3.0 / 56.0 * xs ** 2 / xl ** 6)
    return drker33


def get_1D_kernels_k20(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker20 = scale*(2 / xl - 2/3 * xs/xl**2)

    return drker20



