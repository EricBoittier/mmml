traj_re = ase_io.read("test.traj", "0:1")


# In[87]:


traj_re


# In[88]:


view_atoms(traj_re[0])


# In[89]:


ase_io.write("test.xyz", traj_re)


# In[90]:


x = np.array([_.get_positions()[:,0] for _ in traj_re])
y = np.array([_.get_positions()[:,1]  for _ in traj_re])
z = np.array([_.get_positions()[:,2]  for _ in traj_re])


# In[103]:


Z = [_.get_chemical_symbols()  for _ in traj_re]


import chemcoord as cc
import time


# In[109]:


df = pd.DataFrame({"atom": Z[0], "x": x.flatten(), "y":y.flatten(), "z": z.flatten()})
df


# In[110]:


cart = cc.Cartesian(df)
cart


# In[111]:


cart.index


# In[113]:


zmat = cart.get_zmat()


# In[114]:


zmat


# In[118]:


dir(zmat)
