libnames = [('os','os'), ('sys', 'sys'),('numpy', 'np'),('torch','torch'),('torch.nn','nn')]
for (name, short) in libnames:
  try:
    lib = __import__(name)
  except ImportError:
    print("Library %s cannot be loaded, exiting" % name)
    exit(0)
  else:
    globals()[short] = lib
from torch import nn
from NNucleate.trainig import test, train, NNCV, CVTrajectory
def PLUNNED2(modelpath,trajectorypath,plumedfile='plumed.dat',colvarname='COLVAR',run='no'):
  tf = open(trajectorypath, "r")
  Noatom=int(tf.readline())
  tf.close()

  model=torch.load(modelpath)
  # model.eval()
  Weights=[]
  Biases=[]
  ActL=[]
  for layer in model.sig_stack.children():
      if isinstance(layer, nn.Linear):
        Weights.append(layer.state_dict()['weight'])
        Biases.append(layer.state_dict()['bias'])
      else:
        ActL.append(str(layer)[:-2])
  Layers=int(len(Weights))-1

  Layernodes=[]
  for iter,Weight in enumerate(Weights):
    Layernodes.append(Weight.size(dim=1))

  def Activation(Act,bias):
    if bias>0.0:
      if Act == 'Relu': printfun = "step(x+%0.6f)*(x+%0.6f)" % (bias,bias)
      elif Act == 'Sigmoid': printfun = "1.0/(1.0+exp(-x-%0.6f))" % (bias)
      elif Act == 'Linear': printfun = "(x+%0.6f)" % (bias)
    else:
      if Act == 'Relu': printfun = "step(x-%0.6f)*(x-%0.6f)" % (-bias,-bias)
      elif Act == 'Sigmoid': printfun = "1.0/(1.0+exp(-x+%0.6f))" % (-bias)
      elif Act == 'Linear': printfun = "(x-%0.6f)" % (-bias)
    return printfun

  maxbox=1

  print("Writing Plumed input into %s" % plumedfile)
  print("")
  # traj = md.load(infilename, top=intopname)
  # table, bonds = traj.topology.to_dataframe()
  # atoms = table['serial'][:]
  ofile = open(plumedfile, "w")
  # ofile.write("WHOLEMOLECULES ENTITY0=1-%i\n" % np.max(atoms))
  # ofile.write("FIT_TO_TEMPLATE STRIDE=1 REFERENCE=%s TYPE=OPTIMAL\n" % intopname)
  for i in range(Noatom):#
    ofile.write("p%i: POSITION ATOM=%i NOPBC\n" % (i+1,i+1))
  for i in range(Noatom):#Normalisation and seperation of axis
    ofile.write("p%ix: COMBINE ARG=p%i.x COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
    ofile.write("p%iy: COMBINE ARG=p%i.y COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
    ofile.write("p%iz: COMBINE ARG=p%i.z COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))

  for i in range(Layernodes[1]):#For each item in layer 1
    toprint = "l1_%i: COMBINE ARG=" % (i+1)
    for j in range(int(Layernodes[0]/3)):#For each atom
      toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
    toprint = toprint[:-1] + " COEFFICIENTS="#-1 gets rid of comma
    for j in range(Layernodes[0]):
      toprint = toprint + "%0.6f," % (Weights[0][i,j])#Iterate through weight file
    toprint = toprint[:-1] + " PERIODIC=NO\n"
    ofile.write(toprint)
  for i in range(Layernodes[1]):
    onebias =Biases[0][i]
    fun=Activation(ActL[0],onebias)
    ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,fun))
  
  for layers in range(2,Layers+1):#Iterate through hidden layers
    for i in range(Layernodes[layers]):
      toprint = "l%i_%i: COMBINE ARG=" % (layers,i+1)
      for j in range(Layernodes[layers-1]):
        toprint = toprint + "l%ir_%i," % (layers-1,j+1)
      toprint = toprint[:-1] + " COEFFICIENTS="
      for j in range(Layernodes[layers-1]):
        toprint = toprint + "%0.6f," % (Weights[layers-1][i,j])
      toprint = toprint[:-1] + " PERIODIC=NO\n"
      ofile.write(toprint)
    for i in range(Layernodes[layers]):
      onebias = Biases[layers-1][i]
      fun=Activation(ActL[layers-1],onebias)
      ofile.write("l%ir_%i: MATHEVAL ARG=l%i_%i FUNC=%s PERIODIC=NO\n" % (layers,i+1,layers,i+1,fun))
  toprint = "l%i: COMBINE ARG=" % (Layers+1)
  for j in range(Layernodes[Layers]):
    toprint = toprint + "l%ir_%i," % (Layers,j+1)
  toprint = toprint[:-1] + " COEFFICIENTS="
  for j in range(Layernodes[Layers]):
    toprint = toprint + "%0.6f," % (Weights[Layers][0,j])
  toprint = toprint[:-1] + " PERIODIC=NO\n"
  ofile.write(toprint)
  if Biases[Layers]>0.0:
    ofile.write("l%ir: MATHEVAL ARG=l%i FUNC=(x+%0.6f) PERIODIC=NO\n" % (Layers+1,Layers+1,Biases[Layers]))
  else:
    ofile.write("l%ir: MATHEVAL ARG=l%i FUNC=(x-%0.6f) PERIODIC=NO\n" % (Layers+1,Layers+1,-Biases[Layers]))
  toprint = "PRINT ARG=l%ir STRIDE=1 FILE=%s\n" % (Layers+1,colvarname)
  ofile.write(toprint)
  ofile.close()
  if run=='yes':
#    Command="plumed driver --plumed %s --ixyz %s" %(plumedfile,trajectorypath)
    Command="plumed driver --plumed %s --ixyz %s --length-units A" %(plumedfile,trajectorypath)
    os.system(Command)
