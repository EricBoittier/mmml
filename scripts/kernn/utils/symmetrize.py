# torch imports
import torch
import torch.nn as nn


def ABCC_sym(k1d, nintsym):
    """
    function that includes permutational invariance for a
    molecular system with ABCC symmetrie (such as H2CO).
    
    Permutational invariance is included using fundamental
    invariants (https://doi.org/10.1063/1.4961454)
    
    The numbering is 
    C:0
    O:1
    H:2
    H:3
    and bond symances
    0: C-O
    1: C-H1
    2: C-H2
    3: O-H2
    4: O-H3
    5: H2-H3
    
    """

    if len(k1d.shape) == 1:
        
        sym = torch.zeros(nintsym+1)
        sym[0] = k1d[0]
        sym[1] = k1d[1] + k1d[2]
        sym[2] = k1d[3] + k1d[4]
        sym[3] = k1d[1]**2 + k1d[2]**2
        sym[4] = k1d[3]**2 + k1d[4]**2
        sym[5] = k1d[1]*k1d[3] + k1d[2]*k1d[4]
        sym[6] = k1d[5]
    elif len(k1d.shape) == 2:
        sym = torch.zeros(k1d.shape[0], nintsym+1)
        sym[:, 0] = k1d[:, 0]
        sym[:, 1] = k1d[:, 1] + k1d[:, 2]
        sym[:, 2] = k1d[:, 3] + k1d[:, 4]
        sym[:, 3] = k1d[:, 1]**2 + k1d[:, 2]**2
        sym[:, 4] = k1d[:, 3]**2 + k1d[:, 4]**2
        sym[:, 5] = k1d[:, 1]*k1d[:, 3] + k1d[:, 2]*k1d[:, 4]
        sym[:, 6] = k1d[:, 5]

    else:
        print("ERROR: Please check that the shape of the 1D Kernel array is correct")
    return sym
    
def acem_sym(k1d, nintsym):
    """
    function that calculates the interatomic symances
    of the acetamide molecule given the cartesian coord.
    This function does not take permutational invariance
    into account.
    
    Permutational invariance is included using fundamental
    invariants (https://doi.org/10.1063/1.4961454)
    
    numbering is 
    C:0
    C:1
    N:2
    H:3
    H:4
    O:5
    H:6
    H:7
    H:8
    and bond symances
    0: C0-C1
    1: C0-N
    2: C0-H3
    3: C0-H4
    4: C0-O
    5: C0-H6
    6: C0-H7
    7: C0-H8
    
    8: C1-N
    9: C1-H3
    10: C1-H4
    11: C1-O
    12: C1-H6
    13: C1-H7
    14: C1-H8
    
    15: N-H3
    16: N-H4
    17: N-O
    18: N-H6
    19: N-H7
    20: N-H8
    
    21: H3-H4
    22: H3-O
    23: H3-H6
    24: H3-H7
    25: H3-H8
    
    26: H4-O
    27: H4-H6
    28: H4-H7
    29: H4-H8
    
    30: O-H6
    31: O-H7
    32: O-H8
    
    33: H6-H7
    34: H6-H8
    
    35:H7-H8
    
    """
    if len(k1d.shape) == 1:
        
        sym = torch.zeros(nintsym)
        
        sym[0] = k1d[33] + k1d[34] + k1d[35]
        sym[1] = k1d[5] + k1d[7] + k1d[6]
        sym[2] = k1d[33]**2 + k1d[34]**2 + k1d[35]**2
        sym[3] = k1d[5]**2 + k1d[7]**2 + k1d[6]**2
        sym[4] = k1d[33]*k1d[5] + k1d[34]*k1d[5] + k1d[34]*k1d[7] + k1d[35]*k1d[7] + k1d[35]*k1d[6] + k1d[33]*k1d[6]
        sym[5] = k1d[33]**3 + k1d[34]**3 + k1d[35]**3
        sym[6] = k1d[5]**3 + k1d[7]**3 + k1d[6]**3
        sym[7] = k1d[33]**2*k1d[5] + k1d[34]**2*k1d[5] + k1d[34]**2*k1d[7] + k1d[35]**2*k1d[7] + k1d[35]**2*k1d[6] + k1d[33]**2*k1d[6]
        sym[8] = k1d[5]**2*k1d[35] + k1d[33]*k1d[7]**2 + k1d[34]*k1d[6]**2
        
        sym[9] = k1d[33] + k1d[34] + k1d[35]  
        sym[10] = k1d[12] + k1d[14] + k1d[13]
        sym[11] = k1d[33]**2 + k1d[34]**2 + k1d[35]**2
        sym[12] = k1d[12]**2 + k1d[14]**2 + k1d[13]**2
        sym[13] = k1d[33]*k1d[12] + k1d[34]*k1d[12] +k1d[34]*k1d[14] + k1d[35]*k1d[14] +k1d[35]*k1d[13] + k1d[33]*k1d[13]
        sym[14] = k1d[33]**3 + k1d[34]**3 + k1d[35]**3
        sym[15] = k1d[12]**3 + k1d[14]**3 + k1d[13]**3
        sym[16] = k1d[33]**2 * k1d[12] + k1d[34]**2 * k1d[12] +k1d[34]**2 * k1d[14] + k1d[35]**2 * k1d[14] +k1d[35]**2 * k1d[13] + k1d[33]**2 * k1d[13]
        sym[17] = k1d[12]**2 * k1d[35] + k1d[33] * k1d[14]**2 +k1d[34] * k1d[13]**2
        
        sym[18] = k1d[33] + k1d[34] + k1d[35]
        sym[19] = k1d[30] + k1d[32] + k1d[31]
        sym[20] = k1d[33]**2 + k1d[34]**2 + k1d[35]**2
        sym[21] = k1d[30]**2 + k1d[32]**2 + k1d[31]**2
        sym[22] = k1d[33]*k1d[30] + k1d[34]*k1d[30] +k1d[34]*k1d[32] + k1d[35]*k1d[32] +k1d[35]*k1d[31] + k1d[33]*k1d[31]
        sym[23] = k1d[33]**3 + k1d[34]**3 + k1d[35]**3
        sym[24] = k1d[30]**3 + k1d[32]**3 + k1d[31]**3
        sym[25] = k1d[33]**2 * k1d[30] + k1d[34]**2 * k1d[30] +k1d[34]**2 * k1d[32] + k1d[35]**2 * k1d[32] +k1d[35]**2 * k1d[31] + k1d[33]**2 * k1d[31]
        sym[26] = k1d[30]**2 * k1d[35] + k1d[33] * k1d[32]**2 +k1d[34] * k1d[31]**2
	    
        sym[27] =k1d[0]
        sym[28] = k1d[1]
        sym[29] = k1d[2]
        sym[30] = k1d[3]
        sym[31] = k1d[4]
        sym[32] = k1d[8]
        sym[33] = k1d[9]
        sym[34] = k1d[10]
        sym[35] = k1d[11]
        sym[36] = k1d[15]
        sym[37] = k1d[16]
        sym[38] = k1d[17]
        sym[39] = k1d[21]
        sym[40] = k1d[22]
        sym[41] = k1d[26]
        
        
    elif len(k1d.shape) == 2:
        sym = torch.zeros(k1d.shape[0], nintsym)
        
        sym[:, 0] = k1d[:, 33] + k1d[:, 34] + k1d[:, 35]
        sym[:, 1] = k1d[:, 5] + k1d[:, 7] + k1d[:, 6]
        sym[:, 2] = k1d[:, 33]**2 + k1d[:, 34]**2 + k1d[:, 35]**2
        sym[:, 3] = k1d[:, 5]**2 + k1d[:, 7]**2 + k1d[:, 6]**2
        sym[:, 4] = k1d[:, 33] * k1d[:, 5] + k1d[:, 34] * k1d[:, 5] + k1d[:, 34] * k1d[:, 7] + k1d[:, 35] * k1d[:, 7] + k1d[:, 35] * k1d[:, 6] + k1d[:, 33] * k1d[:, 6]
        sym[:, 5] = k1d[:, 33]**3 + k1d[:, 34]**3 + k1d[:, 35]**3
        sym[:, 6] = k1d[:, 5]**3 + k1d[:, 7]**3 + k1d[:, 6]**3
        sym[:, 7] = k1d[:, 33]**2 * k1d[:, 5] + k1d[:, 34]**2 * k1d[:, 5] + k1d[:, 34]**2 * k1d[:, 7] + k1d[:, 35]**2 * k1d[:, 7] + k1d[:, 35]**2 * k1d[:, 6] + k1d[:, 33]**2 * k1d[:, 6]
        sym[:, 8] = k1d[:, 5]**2 * k1d[:, 35] + k1d[:, 33] * k1d[:, 7]**2 + k1d[:, 34] * k1d[:, 6]**2

        sym[:, 9] = k1d[:, 33] + k1d[:, 34] + k1d[:, 35]
        sym[:, 10] = k1d[:, 12] + k1d[:, 14] + k1d[:, 13]
        sym[:, 11] = k1d[:, 33]**2 + k1d[:, 34]**2 + k1d[:, 35]**2
        sym[:, 12] = k1d[:, 12]**2 + k1d[:, 14]**2 + k1d[:, 13]**2
        sym[:, 13] = k1d[:, 33] * k1d[:, 12] + k1d[:, 34] * k1d[:, 12] + k1d[:, 34] * k1d[:, 14] + k1d[:, 35] * k1d[:, 14] + k1d[:, 35] * k1d[:, 13] + k1d[:, 33] * k1d[:, 13]
        sym[:, 14] = k1d[:, 33]**3 + k1d[:, 34]**3 + k1d[:, 35]**3
        sym[:, 15] = k1d[:, 12]**3 + k1d[:, 14]**3 + k1d[:, 13]**3
        sym[:, 16] = k1d[:, 33]**2 * k1d[:, 12] + k1d[:, 34]**2 * k1d[:, 12] + k1d[:, 34]**2 * k1d[:, 14] + k1d[:, 35]**2 * k1d[:, 14] + k1d[:, 35]**2 * k1d[:, 13] + k1d[:, 33]**2 * k1d[:, 13]
        sym[:, 17] = k1d[:, 12]**2 * k1d[:, 35] + k1d[:, 33] * k1d[:, 14]**2 + k1d[:, 34] * k1d[:, 13]**2

        sym[:, 18] = k1d[:, 33] + k1d[:, 34] + k1d[:, 35]
        sym[:, 19] = k1d[:, 30] + k1d[:, 32] + k1d[:, 31]
        sym[:, 20] = k1d[:, 33]**2 + k1d[:, 34]**2 + k1d[:, 35]**2
        sym[:, 21] = k1d[:, 30]**2 + k1d[:, 32]**2 + k1d[:, 31]**2
        sym[:, 22] = k1d[:, 33] * k1d[:, 30] + k1d[:, 34] * k1d[:, 30] + k1d[:, 34] * k1d[:, 32] + k1d[:, 35] * k1d[:, 32] + k1d[:, 35] * k1d[:, 31] + k1d[:, 33] * k1d[:, 31]
        sym[:, 23] = k1d[:, 33]**3 + k1d[:, 34]**3 + k1d[:, 35]**3
        sym[:, 24] = k1d[:, 30]**3 + k1d[:, 32]**3 + k1d[:, 31]**3
        sym[:, 25] = k1d[:, 33]**2 * k1d[:, 30] + k1d[:, 34]**2 * k1d[:, 30] + k1d[:, 34]**2 * k1d[:, 32] + k1d[:, 35]**2 * k1d[:, 32] + k1d[:, 35]**2 * k1d[:, 31] + k1d[:, 33]**2 * k1d[:, 31]
        sym[:, 26] = k1d[:, 30]**2 * k1d[:, 35] + k1d[:, 33] * k1d[:, 32]**2 + k1d[:, 34] * k1d[:, 31]**2

        sym[:, 27] = k1d[:, 0]
        sym[:, 28] = k1d[:, 1]
        sym[:, 29] = k1d[:, 2]
        sym[:, 30] = k1d[:, 3]
        sym[:, 31] = k1d[:, 4]
        sym[:, 32] = k1d[:, 8]
        sym[:, 33] = k1d[:, 9]
        sym[:, 34] = k1d[:, 10]
        sym[:, 35] = k1d[:, 11]
        sym[:, 36] = k1d[:, 15]
        sym[:, 37] = k1d[:, 16]
        sym[:, 38] = k1d[:, 17]
        sym[:, 39] = k1d[:, 21]
        sym[:, 40] = k1d[:, 22]
        sym[:, 41] = k1d[:, 26]

    else:
        print("ERROR: Please check that the shape of the 1D Kernel array is correct")
    return sym
   

