# torch imports
import torch
import torch.nn as nn



def get_bond_length_ABA(pos, nintdist):
    """
    function that calculates the interatomic distances
    of a molecule with ABA symmetry (such as HeH2+ or H2O)
    This function does not take permutational invariance
    into account.
    
    the numbering is (for the HeH2+ case)
    H:  0
    He: 1
    H:  2
    
    and bond distances
    0: H1-He
    1: H2-He
    2: H1-H2
    
    """
    if len(pos.shape) == 2:
        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[2, :])
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist


def get_bond_length_ABA_sym(pos, nintdist):
    """
    function that calculates the interatomic distances
    of a molecule with ABA symmetry (such as HeH2+ or H2O)
    This function does not take permutational invariance
    into account.
    
    the numbering is (for the HeH2+ case)
    H:  0
    He: 1
    H:  2
    
    and bond distances
    0: H1-He
    1: H2-He
    2: H1-H2
    
    """
    if len(pos.shape) == 2:
        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :]) + torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[1, :])**2 + torch.linalg.norm(pos[1, :] - pos[2, :])**2
        dist[2] = torch.linalg.norm(pos[0, :] - pos[2, :])
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1) + torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)**2 + torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)**2
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist
    
    

def get_bond_length_ABCC(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the h2co molecule given the cartesian coord.
    This function does not take permutational invariance
    into account.
    
    numbering is 
    C:0
    O:1
    H:2
    H:3
    and bond distances
    0: C-O
    1: C-H1
    2: C-H2
    3: O-H2
    4: O-H3
    5: H2-H3
    
    """
    if len(pos.shape) == 2:

        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[3, :])
        dist[3] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[4] = torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[5] = torch.linalg.norm(pos[2, :] - pos[3, :])
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 4] = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 5] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist


def get_bond_length_ABCC_sym(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the h2co molecule given the cartesian coord.
    
    Permutational invariance is included using fundamental
    invariants (https://doi.org/10.1063/1.4961454)
    
    The numbering is 
    C:0
    O:1
    H:2
    H:3
    and bond distances
    0: C-O
    1: C-H1
    2: C-H2
    3: O-H2
    4: O-H3
    5: H2-H3
    
    """
    if len(pos.shape) == 2:

        dist = torch.zeros(nintdist+1)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :]) + torch.linalg.norm(pos[0, :] - pos[3, :])
        dist[2] = torch.linalg.norm(pos[1, :] - pos[2, :]) + torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[3] = torch.linalg.norm(pos[0, :] - pos[2, :])**2 + torch.linalg.norm(pos[0, :] - pos[3, :])**2
        dist[4] = torch.linalg.norm(pos[1, :] - pos[2, :])**2 + torch.linalg.norm(pos[1, :] - pos[3, :])**2
        dist[5] = torch.linalg.norm(pos[0, :] - pos[2, :])*torch.linalg.norm(pos[1, :] - pos[2, :]) + torch.linalg.norm(pos[0, :] - pos[3, :])*torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[6] = torch.linalg.norm(pos[2, :] - pos[3, :])
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist+1)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1) + torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1) + torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)**2 + torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)**2
        dist[:, 4] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)**2 + torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)**2
        dist[:, 5] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)*torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1) + torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)*torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 6] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)

    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist
    
    

def get_bond_length_hoxa(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the  molecule given the cartesian coord.
    numbering is 
    C:0
    C:1
    O:2
    O:3
    O:4
    O:5
    H:6
    and bond distances
    0: C0-C1
    1: C0-O2
    2: C0-O3
    3: C0-O4
    4: C0-O5
    5: C0-H
    
    6: C1-O2
    7: C1-O3    
    8: C1-O4    
    9: C1-O5   
   10: C1-H
   
   11: O2-O3
   12: O2-O4
   13: O2-O5
   14: O2-H
   
   15: O3-O4
   16: O3-O5
   17: O3-H
   
   18: O4-O5
   19: O4-H
    
   20: O5-H
    
    """
    if len(pos.shape) == 2:

        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[3, :])
        dist[3] = torch.linalg.norm(pos[0, :] - pos[4, :])
        dist[4] = torch.linalg.norm(pos[0, :] - pos[5, :])
        dist[5] = torch.linalg.norm(pos[0, :] - pos[6, :])
        
        dist[6] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[7] = torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[8] = torch.linalg.norm(pos[1, :] - pos[4, :])
        dist[9] = torch.linalg.norm(pos[1, :] - pos[5, :])
        dist[10] = torch.linalg.norm(pos[1, :] - pos[6, :])
        
        dist[11] = torch.linalg.norm(pos[2, :] - pos[3, :])
        dist[12] = torch.linalg.norm(pos[2, :] - pos[4, :])
        dist[13] = torch.linalg.norm(pos[2, :] - pos[5, :])
        dist[14] = torch.linalg.norm(pos[2, :] - pos[6, :])
        
        dist[15] = torch.linalg.norm(pos[3, :] - pos[4, :])
        dist[16] = torch.linalg.norm(pos[3, :] - pos[5, :])
        dist[17] = torch.linalg.norm(pos[3, :] - pos[6, :])
        
        dist[18] = torch.linalg.norm(pos[4, :] - pos[5, :])
        dist[19] = torch.linalg.norm(pos[4, :] - pos[6, :])

        dist[20] = torch.linalg.norm(pos[5, :] - pos[6, :])
        
        
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 0, :] - pos[:, 4, :], axis=1)
        dist[:, 4] = torch.linalg.norm(pos[:, 0, :] - pos[:, 5, :], axis=1)
        dist[:, 5] = torch.linalg.norm(pos[:, 0, :] - pos[:, 6, :], axis=1)

        dist[:, 6] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 7] = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 8] = torch.linalg.norm(pos[:, 1, :] - pos[:, 4, :], axis=1)
        dist[:, 9] = torch.linalg.norm(pos[:, 1, :] - pos[:, 5, :], axis=1)
        dist[:, 10] = torch.linalg.norm(pos[:, 1, :] - pos[:, 6, :], axis=1)
        
        dist[:, 11] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)
        dist[:, 12] = torch.linalg.norm(pos[:, 2, :] - pos[:, 4, :], axis=1)
        dist[:, 13] = torch.linalg.norm(pos[:, 2, :] - pos[:, 5, :], axis=1)
        dist[:, 14] = torch.linalg.norm(pos[:, 2, :] - pos[:, 6, :], axis=1)
        
        dist[:, 15] = torch.linalg.norm(pos[:, 3, :] - pos[:, 4, :], axis=1)
        dist[:, 16] = torch.linalg.norm(pos[:, 3, :] - pos[:, 5, :], axis=1)
        dist[:, 17] = torch.linalg.norm(pos[:, 3, :] - pos[:, 6, :], axis=1)
        
        dist[:, 18] = torch.linalg.norm(pos[:, 4, :] - pos[:, 5, :], axis=1)
        dist[:, 19] = torch.linalg.norm(pos[:, 4, :] - pos[:, 6, :], axis=1)
        
        dist[:, 20] = torch.linalg.norm(pos[:, 5, :] - pos[:, 6, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist
    
    
    


def get_bond_length_acem(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the acetamide molecule given the cartesian coord.
    This function does not take permutational invariance
    into account.
    
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
    and bond distances
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
    if len(pos.shape) == 2:

        dist = torch.zeros(nintdist)
        dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
        dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :])
        dist[2] = torch.linalg.norm(pos[0, :] - pos[3, :])
        dist[3] = torch.linalg.norm(pos[0, :] - pos[4, :])
        dist[4] = torch.linalg.norm(pos[0, :] - pos[5, :])
        dist[5] = torch.linalg.norm(pos[0, :] - pos[6, :])
        dist[6] = torch.linalg.norm(pos[0, :] - pos[7, :])
        dist[7] = torch.linalg.norm(pos[0, :] - pos[8, :])
                
        dist[8] = torch.linalg.norm(pos[1, :] - pos[2, :])
        dist[9] = torch.linalg.norm(pos[1, :] - pos[3, :])
        dist[10] = torch.linalg.norm(pos[1, :] - pos[4, :])
        dist[11] = torch.linalg.norm(pos[1, :] - pos[5, :])
        dist[12] = torch.linalg.norm(pos[1, :] - pos[6, :])
        dist[13] = torch.linalg.norm(pos[1, :] - pos[7, :])
        dist[14] = torch.linalg.norm(pos[1, :] - pos[8, :])
        
        dist[15] = torch.linalg.norm(pos[2, :] - pos[3, :])
        dist[16] = torch.linalg.norm(pos[2, :] - pos[4, :])
        dist[17] = torch.linalg.norm(pos[2, :] - pos[5, :])
        dist[18] = torch.linalg.norm(pos[2, :] - pos[6, :])
        dist[19] = torch.linalg.norm(pos[2, :] - pos[7, :])
        dist[20] = torch.linalg.norm(pos[2, :] - pos[8, :])
        
        dist[21] = torch.linalg.norm(pos[3, :] - pos[4, :])
        dist[22] = torch.linalg.norm(pos[3, :] - pos[5, :])
        dist[23] = torch.linalg.norm(pos[3, :] - pos[6, :])
        dist[24] = torch.linalg.norm(pos[3, :] - pos[7, :])
        dist[25] = torch.linalg.norm(pos[3, :] - pos[8, :])
        
        dist[26] = torch.linalg.norm(pos[4, :] - pos[5, :])
        dist[27] = torch.linalg.norm(pos[4, :] - pos[6, :])
        dist[28] = torch.linalg.norm(pos[4, :] - pos[7, :])
        dist[29] = torch.linalg.norm(pos[4, :] - pos[8, :])

        dist[30] = torch.linalg.norm(pos[5, :] - pos[6, :])  
        dist[31] = torch.linalg.norm(pos[5, :] - pos[7, :])
        dist[32] = torch.linalg.norm(pos[5, :] - pos[8, :])
        
        dist[33] = torch.linalg.norm(pos[6, :] - pos[7, :])
        dist[34] = torch.linalg.norm(pos[6, :] - pos[8, :])
        
        dist[35] = torch.linalg.norm(pos[7, :] - pos[8, :]) 
        
    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 0, :] - pos[:, 4, :], axis=1)
        dist[:, 4] = torch.linalg.norm(pos[:, 0, :] - pos[:, 5, :], axis=1)
        dist[:, 5] = torch.linalg.norm(pos[:, 0, :] - pos[:, 6, :], axis=1)
        dist[:, 6] = torch.linalg.norm(pos[:, 0, :] - pos[:, 7, :], axis=1)
        dist[:, 7] = torch.linalg.norm(pos[:, 0, :] - pos[:, 8, :], axis=1)

        dist[:, 8] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 9] = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 10] = torch.linalg.norm(pos[:, 1, :] - pos[:, 4, :], axis=1)
        dist[:, 11] = torch.linalg.norm(pos[:, 1, :] - pos[:, 5, :], axis=1)
        dist[:, 12] = torch.linalg.norm(pos[:, 1, :] - pos[:, 6, :], axis=1)
        dist[:, 13] = torch.linalg.norm(pos[:, 1, :] - pos[:, 7, :], axis=1)
        dist[:, 14] = torch.linalg.norm(pos[:, 1, :] - pos[:, 8, :], axis=1)
        
        dist[:, 15] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)
        dist[:, 16] = torch.linalg.norm(pos[:, 2, :] - pos[:, 4, :], axis=1)
        dist[:, 17] = torch.linalg.norm(pos[:, 2, :] - pos[:, 5, :], axis=1)
        dist[:, 18] = torch.linalg.norm(pos[:, 2, :] - pos[:, 6, :], axis=1)
        dist[:, 19] = torch.linalg.norm(pos[:, 2, :] - pos[:, 7, :], axis=1)
        dist[:, 20] = torch.linalg.norm(pos[:, 2, :] - pos[:, 8, :], axis=1)
        
        dist[:, 21] = torch.linalg.norm(pos[:, 3, :] - pos[:, 4, :], axis=1)
        dist[:, 22] = torch.linalg.norm(pos[:, 3, :] - pos[:, 5, :], axis=1)
        dist[:, 23] = torch.linalg.norm(pos[:, 3, :] - pos[:, 6, :], axis=1)
        dist[:, 24] = torch.linalg.norm(pos[:, 3, :] - pos[:, 7, :], axis=1)
        dist[:, 25] = torch.linalg.norm(pos[:, 3, :] - pos[:, 8, :], axis=1)
        
        dist[:, 26] = torch.linalg.norm(pos[:, 4, :] - pos[:, 5, :], axis=1)
        dist[:, 27] = torch.linalg.norm(pos[:, 4, :] - pos[:, 6, :], axis=1)
        dist[:, 28] = torch.linalg.norm(pos[:, 4, :] - pos[:, 7, :], axis=1)
        dist[:, 29] = torch.linalg.norm(pos[:, 4, :] - pos[:, 8, :], axis=1)
        
        dist[:, 30] = torch.linalg.norm(pos[:, 5, :] - pos[:, 6, :], axis=1)
        dist[:, 31] = torch.linalg.norm(pos[:, 5, :] - pos[:, 7, :], axis=1)
        dist[:, 32] = torch.linalg.norm(pos[:, 5, :] - pos[:, 8, :], axis=1)
        
        dist[:, 33] = torch.linalg.norm(pos[:, 6, :] - pos[:, 7, :], axis=1)
        dist[:, 34] = torch.linalg.norm(pos[:, 6, :] - pos[:, 8, :], axis=1)
        
        dist[:, 35] = torch.linalg.norm(pos[:, 7, :] - pos[:, 8, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist


def get_bond_length_form(pos, nintdist):
    """
    Calculate all interatomic distances for FORM
    with atom indexing:
    H:0, C:1, N:2, H:3, H:4, O:5

    Bond-distance numbering:
    0:  H0-C
    1:  H0-N
    2:  H0-H3
    3:  H0-H4
    4:  H0-O
    5:  C-N
    6:  C-H3
    7:  C-H4
    8:  C-O
    9:  N-H3
    10: N-H4
    11: N-O
    12: H3-H4
    13: H3-O
    14: H4-O
    """

    if len(pos.shape) == 2:
        dist = torch.zeros(nintdist)

        dist[0]  = torch.linalg.norm(pos[0, :] - pos[1, :])  # H0-C
        dist[1]  = torch.linalg.norm(pos[0, :] - pos[2, :])  # H0-N
        dist[2]  = torch.linalg.norm(pos[0, :] - pos[3, :])  # H0-H3
        dist[3]  = torch.linalg.norm(pos[0, :] - pos[4, :])  # H0-H4
        dist[4]  = torch.linalg.norm(pos[0, :] - pos[5, :])  # H0-O
        dist[5]  = torch.linalg.norm(pos[1, :] - pos[2, :])  # C-N
        dist[6]  = torch.linalg.norm(pos[1, :] - pos[3, :])  # C-H3
        dist[7]  = torch.linalg.norm(pos[1, :] - pos[4, :])  # C-H4
        dist[8]  = torch.linalg.norm(pos[1, :] - pos[5, :])  # C-O
        dist[9]  = torch.linalg.norm(pos[2, :] - pos[3, :])  # N-H3
        dist[10] = torch.linalg.norm(pos[2, :] - pos[4, :])  # N-H4
        dist[11] = torch.linalg.norm(pos[2, :] - pos[5, :])  # N-O
        dist[12] = torch.linalg.norm(pos[3, :] - pos[4, :])  # H3-H4
        dist[13] = torch.linalg.norm(pos[3, :] - pos[5, :])  # H3-O
        dist[14] = torch.linalg.norm(pos[4, :] - pos[5, :])  # H4-O

    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)

        dist[:, 0]  = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)  
        dist[:, 1]  = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)  
        dist[:, 2]  = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)  
        dist[:, 3]  = torch.linalg.norm(pos[:, 0, :] - pos[:, 4, :], axis=1)  
        dist[:, 4]  = torch.linalg.norm(pos[:, 0, :] - pos[:, 5, :], axis=1)  
        dist[:, 5]  = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)  
        dist[:, 6]  = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)  
        dist[:, 7]  = torch.linalg.norm(pos[:, 1, :] - pos[:, 4, :], axis=1)  
        dist[:, 8]  = torch.linalg.norm(pos[:, 1, :] - pos[:, 5, :], axis=1) 
        dist[:, 9]  = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1) 
        dist[:, 10] = torch.linalg.norm(pos[:, 2, :] - pos[:, 4, :], axis=1)  
        dist[:, 11] = torch.linalg.norm(pos[:, 2, :] - pos[:, 5, :], axis=1)  
        dist[:, 12] = torch.linalg.norm(pos[:, 3, :] - pos[:, 4, :], axis=1) 
        dist[:, 13] = torch.linalg.norm(pos[:, 3, :] - pos[:, 5, :], axis=1) 
        dist[:, 14] = torch.linalg.norm(pos[:, 4, :] - pos[:, 5, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist


def get_bond_length_ABBCC(pos, nintdist):
    """
    function that calculates the interatomic distances
    of the h2co molecule given the cartesian coord.
    
    Permutational invariance is included using fundamental
    invariants (https://doi.org/10.1063/1.4961454)
    
    The numbering is 
    C:0
    O:1
    O:2
    H:3
    H:4
    and bond distances
    0: C-O1
    1: C-O2
    2: C-H1
    3: O-H2
    4: O1-O2
    5: O1-H1
    6: O1-H2
    7: O2-H1
    8: O2-H2
    9: H1-H2
    
    """
    if len(pos.shape) == 2:
    
    	dist = torch.zeros(nintdist)
    	dist[0] = torch.linalg.norm(pos[0, :] - pos[1, :])
    	dist[1] = torch.linalg.norm(pos[0, :] - pos[2, :])
    	dist[2] = torch.linalg.norm(pos[0, :] - pos[3, :])
    	dist[3] = torch.linalg.norm(pos[0, :] - pos[4, :])
    	dist[4] = torch.linalg.norm(pos[1, :] - pos[2, :])
    	dist[5] = torch.linalg.norm(pos[1, :] - pos[3, :])
    	dist[6] = torch.linalg.norm(pos[1, :] - pos[4, :])
    	dist[7] = torch.linalg.norm(pos[2, :] - pos[3, :])
    	dist[8] = torch.linalg.norm(pos[2, :] - pos[4, :])
    	dist[9] = torch.linalg.norm(pos[3, :] - pos[4, :])


    elif len(pos.shape) == 3:
        dist = torch.zeros(pos.shape[0], nintdist)
        dist[:, 0] = torch.linalg.norm(pos[:, 0, :] - pos[:, 1, :], axis=1)
        dist[:, 1] = torch.linalg.norm(pos[:, 0, :] - pos[:, 2, :], axis=1)
        dist[:, 2] = torch.linalg.norm(pos[:, 0, :] - pos[:, 3, :], axis=1)
        dist[:, 3] = torch.linalg.norm(pos[:, 0, :] - pos[:, 4, :], axis=1)
        dist[:, 4] = torch.linalg.norm(pos[:, 1, :] - pos[:, 2, :], axis=1)
        dist[:, 5] = torch.linalg.norm(pos[:, 1, :] - pos[:, 3, :], axis=1)
        dist[:, 6] = torch.linalg.norm(pos[:, 1, :] - pos[:, 4, :], axis=1)
        dist[:, 7] = torch.linalg.norm(pos[:, 2, :] - pos[:, 3, :], axis=1)
        dist[:, 8] = torch.linalg.norm(pos[:, 2, :] - pos[:, 4, :], axis=1)
        dist[:, 9] = torch.linalg.norm(pos[:, 3, :] - pos[:, 4, :], axis=1)
    else:
        print("ERROR: Please check that the shape of the position array is correct")
    return dist
    
    


