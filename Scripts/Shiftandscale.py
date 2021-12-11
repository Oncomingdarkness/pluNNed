import numpy as np
import pandas as pd

def shift(trajectory,box_length):
  def transform(x,box_length):
    while(x >= box_length):
        x -= box_length
    while(x <= 0):
        x += box_length
    return x/(box_length/10)
  def conv(x):
    out=str(round(x,5))
    return out    
  func=np.vectorize(transform)
  flo=np.vectorize(float)
  string=np.vectorize(conv)
  file = open(traj, "r")
  Content=file.readlines()
  file.close()
  Atoms=int(Content[0])
  Step=Atoms+2
  Frames=int(len(Content)/(Atoms+2))
  # Frames=10
  tofile=[]
  for i in range(Frames):
    tofile.append(Content[Step*i])
    tofile.append(Content[Step*i+1])
    for j in range(Atoms):
      Line=list(string(func(flo(Content[2+Step*i+j].split()[1:4]),box_length)))
      Line.insert (0, str(1))
      Line1=" ".join(Line)
      Line1+="\n"
      tofile.append(Line1)
  substr=".xyz"
  idx = trajectory.index(substr)
  filename=trajectory[:idx] + "_DivShift" + trajectory[idx:]
  outf = open(filename, "w")
  outf.writelines(tofile)
  outf.close()
