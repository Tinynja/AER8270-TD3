# Standard libs
from math import *
import multiprocessing
from functools import partial
from itertools import product

# Pip libs
import numpy as np
import matplotlib.pyplot as plt

# User libs
from libraries.vlm import *

# Question 1 (a)
def run_q1a():
	# Run VLM
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
	# Print results
	print("----Question 1 (a)----")
	print("Pente de portance-incidence: %f*pi" % (q1a.CL_alpha/pi))
	print("Trainee: %e" % q1a.CD[-1])
	# Return results
	return q1a

# Question 1 (b)
def vlm_enqueue(queue, sweep, ar):
	vlm_prob = VLM(ni=5,
				   nj=50,
				   chordRoot=1.0,
				   chordTip=1.0,
				   twistRoot=0.0,
				   twistTip=0.0,
				   span=ar,
				   sweep=sweep,
				   Sref=ar,
				   referencePoint=[0.0,0.0,0.0],
				   wingType=1,
				   alphaRange = [0,10])
	vlm_prob.run()
	queue.put(['probs', sweep, ar, vlm_prob])
	queue.put(['CL_alpha', sweep, ar, vlm_prob.CL_alpha])

def run_q1b(pool, queue):
	# Defining the aspect ratio and CL_alpha lists
	q1b = {'ar':[1e-5]+list(np.arange(0.5,7.5,0.5)), 'sweep':[0, 30, 45, 60], 'probs':[], 'CL_alpha':[]}
	# Hierarchy: probs/sweep/ar
	for i in range(len(q1b['sweep'])):
		q1b['probs'].append([0]*len(q1b['ar']))
		q1b['CL_alpha'].append([0]*len(q1b['ar']))

	# Enquing VLM results from multiprocessor pool for each combination of a/r and sweep
	vlm_partial = partial(vlm_enqueue, queue)
	pool.starmap(vlm_partial, product(q1b['sweep'], q1b['ar']))

	# Dequeue results
	for i in range(q.qsize()):
		qi = q.get()
		q1b[qi[0]][q1b['sweep'].index(qi[1])][q1b['ar'].index(qi[2])] = qi[3]

	# Generate plot
	for i in range(len(q1b['sweep'])):
		plt.plot(q1b['ar'], q1b['CL_alpha'][i])
	plt.xlabel('Aspect ratio AR')
	plt.ylabel('CL_alpha')
	plt.title("Effet de l'aspect ratio et de la sweep sur la pente CL_alpha")
	plt.grid()
	plt.savefig('figs/q1b.png') #save plot to file

	return q1b

if __name__ == '__main__':
	multiprocessing.freeze_support()
	p = multiprocessing.Pool()
	q = multiprocessing.Manager().Queue()
	#q1a = run_q1a()
	q1b = run_q1b(p, q)