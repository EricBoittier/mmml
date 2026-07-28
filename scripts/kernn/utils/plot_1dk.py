#!/usr/bin/env python

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch

from utils.distances import get_bond_length_acem
from utils.kernels import get_1D_kernels_k20
from utils.symmetrize import acem_sym

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 28

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, ax = plt.subplots()

[x.set_linewidth(1.5) for x in ax.spines.values()]

#get rid of all the borders except bottm x axis.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.ylabel('$k^{[2,0]}\;\,(r, r\')$', fontsize=28)
plt.xlabel('$r \;\, \mathrm{(Å)}$', fontsize=28)

npoints = 1000

rprimes = torch.FloatTensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])

n = len(rprimes)


r = torch.linspace(0.4, 10, npoints)


# Define colormap (continuous)
cmap = plt.get_cmap('RdYlBu_r', n)


# Loop to create and plot each line

cc = 0
for rprime in rprimes:
    k = get_1D_kernels_k20(r, rprime) 
    ax.plot(r, k, color=cmap(cc), label='$r\' = '+str(rprime)+' \;\, \mathrm{(Å)}$')
    cc += 1



plt.tight_layout()
plt.legend(ncols=2)



#plt.tight_layout()
#plt.savefig("oneD-kernels_33.png",bbox_inches='tight',dpi=300)
plt.show()
quit()


