# Standard libs
from math import *

# Pip libs
import numpy as np
import matplotlib.pyplot as plt

# User libs
from libs.vlm import *

# Question 1 (a)
q1a = VLM(ni=5,
	     nj=50,
	     chordRoot=1,
	     chordTip=1,
	     twistRoot=0.0,
	     twistTip=0.0,
	     span=1e5,
	     sweep=0.0,
	     Sref =1e5,
	     referencePoint=[0.0,0.0,0.0],
	     wingType=1,
	     alphaRange = [0,10])

q1a.run()

print("----Question 1 (a)----")
print("Pente de portance-incidence: %f*pi" % (q1a.CL_alpha/pi))
print("Trainee: %e" % q1a.CD[-1])

# Question 1 (b)
if 1:
	print("----Question 1 (b)----")

	q1b = {'probs':[], 'ar':[1e-5]+list(np.arange(0.5,8,0.5)), 'CL_alpha':[]}

	for i in q1b['ar']:
		print("Calculating aspect ratio: %f" % i)
		q1b['probs'].append(VLM(ni=5,
							    nj=50,
							    chordRoot=1.0,
							    chordTip=1.0,
							    twistRoot=0.0,
							    twistTip=0.0,
							    span=i,
							    sweep=0.0,
							    Sref=i,
							    referencePoint=[0.0,0.0,0.0],
							    wingType=1,
							    alphaRange = [0,10]))
		q1b['probs'][-1].run()
		q1b['CL_alpha'].append(q1b['probs'][-1].CL_alpha)

	# Generate plot
	plt.plot(q1b['ar'], q1b['CL_alpha'])
	plt.xlabel('Aspect ratio AR')
	plt.ylabel('CL_alpha')
	plt.title("Effet de l'aspect ratio et de la fleche sur la pente CL_alpha")
	plt.grid()
	plt.savefig('figs/q1b.png') #save plot to file