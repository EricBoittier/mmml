import numpy as np

mp2_data = np.load('hydrogen_oxalate_mp2_avtz_gen2_22200.npz')
cc_data = np.load('hydrogen_oxalate_TL_cc_avtz_2688.npz')


print(mp2_data.keys())
print(cc_data.keys())

print(mp2_data['R'].shape)
print(cc_data['R'].shape)