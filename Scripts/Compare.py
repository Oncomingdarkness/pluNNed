import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

traj_path ='../trajectories/shifted_scaled_Trunc_13change.xyz'
cv_path ='../trajectories/cn.13.dat'
top_path ='../trajectories/reference_13.pdb'
nnpath='../512x256x128correct.pt'
plumed_path='./All'

NN_value=np.loadtxt("NNresults.dat")
Plumed_value=np.loadtxt(plumed_path)[:, 1]
True_value=np.loadtxt(cv_path)[:, 3]
print(NN_value)
print(Plumed_value)
print(True_value)

plt.figure(200)
plt.scatter(Plumed_value,NN_value,s=0.1)
plt.xlabel("PluNNed Predictions")
plt.ylabel("Torch NN Predictions")

plt.figure(300)
plt.xlabel("PluNNed Predictions")
plt.ylabel("True Values")
plt.scatter(Plumed_value, True_value, s=0.1)

plt.show()
