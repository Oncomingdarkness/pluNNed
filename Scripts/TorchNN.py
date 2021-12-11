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

#traj_path ='./trajectories/shifted_scaled_Trunc_13change.xyz'
#cv_path ='./trajectories/cn.13.dat'
#traj_path ='../trajectories/shifted_scaled_Trunc_13change.xyz'
#cv_path ='../trajectories/cn.13trunc.dat'
traj_path ='../trajectories/dump.3Dcolloid.13_DivShift.xyz'
cv_path ='../trajectories/cn.13.dat'
top_path ='../trajectories/reference_13.pdb'
nnpath='../512x256x128correct.pt'
model=torch.load(nnpath)
#ds = CVTrajectory(cv_path, traj_path, top_path, 3, 9.283)
ds = CVTrajectory(cv_path, traj_path, top_path, 3, 1)

val_dataloader1 = DataLoader(ds, batch_size=1, shuffle=False)
index=0
ys = []
preds = []
for batch, (X, y) in enumerate(val_dataloader1):#Should probably be val_dataloader
#  if batch==index:
      Xindex=X
      yindex=y
#      print(X,y)
      model.eval()
      #optimizer.zero_grad()
      pred = model(X)
      #print(pred)
      [ys.append(ref.item()) for ref in y]
      [preds.append(pre.item()) for pre in pred]
#print(ys)
#print(preds)

predstr=[]
for el in range(len(preds)):
  predstr.append(str(preds[el])+"\n")
  fpreds = open("NNresults.dat", "w")
fpreds.writelines(predstr)
fpreds.close()
