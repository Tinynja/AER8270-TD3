# Standard libs
from math import *

# User libs
from libs.vlm import *

# Question 1
q1 = VLM(ni=5,
	     nj=50,
	     chordRoot=1.0,
	     chordTip=1.0,
	     twistRoot=0.0,
	     twistTip=0.0,
	     span=100000,
	     sweep=0.0,
	     Sref =100000,
	     referencePoint=[0.0,0.0,0.0],
	     wingType=1,
	     alphaRange = [0,10])

q1.run()

print("Question 1 (a)")
print("-------------------")
print("Pente de portance-incidence: %f*pi" % ((q1.CL[-1]-q1.CL[0])/radians(q1.alphaRange[-1]-q1.alphaRange[0])/pi))
print("Trainee: %e" % q1.CD[-1])