from vlm import *
from math import *
import numpy as np
a=2 #demi-grand axe
b = 20 #envergure et demi petit axe
Sref = (a/2)*(b/2)*math.pi
AR = b**2/Sref
prob = VLM(ni=5,
		       nj=50,
		       chordRoot=1.0,#aile rectangulaire à la base
		       chordTip=0, #bout d'une aile elliptique
		       twistRoot=0.0,
		       twistTip=0.0,
		       span=b/2,
		       sweep=0.0,
		       Sref =Sref/2,
		       referencePoint=[0.0,0.0,0.0],
		       wingType=3,
		       alphaRange = [0.0,10.0])
prob.run()

[y,cl_y,corde] = prob.lifting_line()

theta=np.zeros(len(y))
gamma = np.zeros(len(cl_y))
An = np.zeros(len(cl_y))
for j in range(0,nj):
	theta[j] = acos(-y[j]/(b/2))
	gamma[j] =0.5*cl_y[j]* corde[j]

#Méthode alternative avec inversion de matrice analytique:
#theta_j=j*pi/r et sin(n*j*pi/r) avec n= (2I+1) car les n sont impairs et j*pi/r=theta(l)) 
    
for i in range (0,nj):
	sigma=0
	for l in range(0,nj):
		sigma = sigma+(gamma[l]*sin((2*i+1)*theta[l]))
	An[i] = (2/(nj+1))*(1/(4*(b/2.0)))*sigma
delta=0
for i in range(1, len(An)):
	delta += (2*i+1)*(An[i]/An[0])**2.0
e = 1.0/(1.0+delta)
CL= math.pi*An[0]*AR
print('facteur Oswald :',e)
Cdi = (CL**2)/(e*math.pi*AR) #NOTE : Cdi a 10deg
print('Cdi :',Cdi)
