from NNucleate.trainig import test, train, NNCV, CVTrajectory
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time
import numpy as np
import pandas as pd
import math
from math import exp

traj_path ='../trajectories/shifted_scaled_Trunc_13change.xyz'
cv_path ='../trajectories/cn.13.dat'
#cv_path ='./trajectories/cn.13trunc.dat'
top_path ='../trajectories/reference_13.pdb'
nnpath='../512x256x128correct.pt'
plumed_path='./All'
#plumed_path='./COLVARtoday'

#model=torch.load(nnpath)

NN_value=np.loadtxt("NNresults.dat")
Plumed_value=np.loadtxt(plumed_path)[:, 1]
True_value=np.loadtxt(cv_path)[:, 3]
print(NN_value)
print(Plumed_value)
print(True_value)
#np.loadtxt(cv_file)[:, cv_col]

# Initialise the subplot function using number of rows and columns
#figure, axis = plt.subplots(1,2)
  
# For Sine Function
#axis[0].scatter(NN_value, Plumed_value, s=0.1)
#axis[0].set_title("Plumed")
  
# For Cosine Function
#axis[1].scatter(NN_value, True_value, s=0.1)
#axis[1].set_title("True")
#plt.show()


plt.figure(200)
plt.scatter(Plumed_value,NN_value,s=0.1)
plt.xlabel("PluNNed Predictions")
plt.ylabel("Torch NN Predictions")
#plt.set_title("Plumed")
plt.figure(300)
plt.xlabel("PluNNed Predictions")
plt.ylabel("True Values")
plt.scatter(Plumed_value, True_value, s=0.1)
#plt.scatter(NN_value, True_value, s=0.1)
plt.show()
