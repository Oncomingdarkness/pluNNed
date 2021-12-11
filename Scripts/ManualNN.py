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
def Activation(Input,Act,Bias):
  if Act=='Relu':Output=max(0,Input+Bias)
  elif Act=='Sigmoid':Output=1/(1+np.exp(-Input-Bias))
  elif Act=='Linear':Output=Input+Bias
  return Output

Weights=[]
Biases=[]
ActL=[]
for layer in model.sig_stack.children():
    if isinstance(layer, nn.Linear):
      Weights.append(layer.state_dict()['weight'])
      Biases.append(layer.state_dict()['bias'])
    else:
      ActL.append(str(layer)[:-2])
Layers=int(len(Weights))

val_dataloader1 = DataLoader(ds, batch_size=1, shuffle=False)

index=0
ys = []
preds = []
Xindex=[]
yindex=[]
for batch, (X, y) in enumerate(val_dataloader1):#Should probably be val_dataloader
#  if batch==index:
      Xindex.append(X)
      yindex.append(y)
#      print(X,y)
      model.eval()
      #optimizer.zero_grad()
      pred = model(X)
      #print(pred)
      [ys.append(ref.item()) for ref in y]
      [preds.append(pre.item()) for pre in pred]
#print(ys)
#print(preds)

Frames=len(yindex)
predictions=[]
for l in range(Frames):
	Xindexflat=torch.flatten(Xindex[l])

	Linear=[]
	Inbetween=[]
	Nonlinear=[]
	Nonlinear.append(Xindexflat)
	for layers in range(Layers-1):
	  Linear.append(np.dot(Weights[layers].numpy(),Nonlinear[layers]))
	  Inbetween.append(Linear[layers]+Biases[layers].numpy())
	  Nonlinear.append(Activation(Linear[layers],ActL[layers],Biases[layers].numpy()))
	Linear.append(np.dot(Weights[Layers-1].numpy(),Nonlinear[Layers-1]))
	Inbetween.append(Linear[Layers-1]+Biases[Layers-1].numpy())
	predictions.append(Inbetween[3])
#print(Inbetween[3])
#print(predictions)

predstr=[]
for el in range(len(predictions)):
  predstr.append(str(predictions[el][0])+"\n")
  fpreds = open("NNmanualresults.dat", "w")
fpreds.writelines(predstr)
fpreds.close()
