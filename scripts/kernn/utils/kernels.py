# torch imports
import torch
import torch.nn as nn




def get_1D_kernels_k20(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker20 = scale*(2 / xl - 2/3 * xs/xl**2)

    return drker20

def get_1D_kernels_k21(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker21 = scale*(2.0 / (3.0 * xl ** 2) - 1.0 / 3.0 * xs / xl ** 3)
    
    return drker21


def get_1D_kernels_k22(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker22 = scale*(1.0 / (3.0 * xl ** 3) - 1.0 / 5.0 * xs / xl ** 4)
    
    return drker22


def get_1D_kernels_k23(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker23 = scale*(1.0 / (5.0 * xl ** 4) - 2.0 / 15.0 * xs / xl ** 5)
    
    return drker23

def get_1D_kernels_k24(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker24 = scale*(2.0 / (15.0 * xl ** 5) - 2.0 / 21.0 * xs / xl ** 6)
    
    return drker24

def get_1D_kernels_k25(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker25 = scale*(2.0 / (21.0 * xl ** 6) - 1.0 / 14.0 * xs / xl ** 7)
    
    return drker25

def get_1D_kernels_k26(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker26 = scale*(1.0 / (14.0 * xl ** 7) - 1.0 / 18.0 * xs / xl ** 8)
    
    return drker26

def get_1D_kernels_k30(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker30 = scale*(3.0 / (xl) - 3.0 / 2.0 * xs / xl ** 2 + 3.0 / 10.0 * xs ** 2 / xl ** 3)
    
    return drker30
    
def get_1D_kernels_k31(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker31 = scale*(3.0 / (4.0 * xl ** 2) - 3.0 / 5.0 * xs / xl ** 3 + 3.0 / 20.0 * xs ** 2 / xl ** 4)
    
    return drker31
    
def get_1D_kernels_k32(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker32 = scale*(3.0 / (10.0 * xl ** 3) - 3.0 / 10.0 * xs / xl ** 4 + 3.0 / 35.0 * xs ** 2 / xl ** 5)
    
    return drker32

def get_1D_kernels_k33(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker33 = scale*(3.0 / (20.0 * xl ** 4) - 6.0 / 35.0 * xs / xl ** 5 + 3.0 / 56.0 * xs ** 2 / xl ** 6)
    
    return drker33
    
def get_1D_kernels_k34(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker34 = scale*(3.0 / (35.0 * xl ** 5) - 3.0 / 28.0 * xs / xl ** 6 + 1.0 / 28.0 * xs ** 2 / xl ** 7)
    
    return drker34
    
def get_1D_kernels_k35(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker35 = scale*(3.0 / (56.0 * xl ** 6) - 1.0 / 14.0 * xs / xl ** 7 + 1.0 / 40.0 * xs ** 2 / xl ** 8)
    
    return drker35
    
def get_1D_kernels_k36(x, xi, scale=1):
    """function calculating the 1D kernel functions given the bond length
    x corresponds to the input, xi is the reference structure)
    """
    xl = torch.maximum(x, xi)
    xs = torch.minimum(x, xi)

    drker36 = scale*(1.0 / (28.0 * xl ** 7) - 1.0 / 20.0 * xs / xl ** 8 + 1.0 / 55.0 * xs ** 2 / xl ** 9)
    
    return drker36

    
    

