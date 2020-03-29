# Standard libs
from math import *
import multiprocessing
import sys
import csv
from functools import partial
from itertools import product

# Pip libs
import numpy as np
import matplotlib.pyplot as plt

# User libs
from libraries.vlm import *


#----Question 1 (a)----
def run_q1a(pool, queue):
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
	# Save results as CSV
	with open("data/q1a.csv", "w", newline="") as f:
		writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(["AR", "CL_alpha", "CD"])
		writer.writerow([q1a.AR, q1a.CL_alpha, q1a.CD[-1]])
	# Print results
	run_q1a_cache()
	# Return results
	return q1a

def run_q1a_cache():
	q1a = []
	# Read results from CSV
	with open("data/q1a.csv", "r", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			q1a = row
	# Print results
	print("----Question 1 (a)----")
	print("Aspect ratio: %g" % q1a["AR"])
	print("Pente de portance-incidence: %f*pi" % (q1a["CL_alpha"]/pi))
	print("Trainee: %e" % q1a["CD"])


#----Question 1 (b)----
def vlm_enqueue(queue, sweep, ar):
	print(f"Calculating (sweep={sweep}, AR={ar*2})...")
	sys.stdout.flush()
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
	queue.put(["probs", sweep, ar, vlm_prob])
	queue.put(["CL_alpha", sweep, ar, vlm_prob.CL_alpha])

def run_q1b(pool, queue):
	# Defining the aspect ratio and CL_alpha lists
	q1b = {"AR":[1e-5]+list(np.arange(0.5,1,0.5)), "sweep":[0, 30], "halfAR": [], "probs":[], "CL_alpha":[]}
	# Hierarchy: probs/sweep/ar
	for i in range(len(q1b["sweep"])):
		q1b["probs"].append([0]*len(q1b["AR"]))
		q1b["CL_alpha"].append([0]*len(q1b["AR"]))
	# Divide AR by 2 to get the AR of a half wing
	for i in range(len(q1b["AR"])):
		q1b["halfAR"].append(q1b["AR"][i]/2)
	# Enquing VLM results from multiprocessor pool for each combination of AR and sweep
	vlm_partial = partial(vlm_enqueue, queue)
	# Divide ar by 2 to get the result for a full wing
	pool.starmap(vlm_partial, product(q1b["sweep"], q1b["halfAR"]))
	# Dequeue results
	for i in range(queue.qsize()):
		qi = queue.get()
		q1b[qi[0]][q1b["sweep"].index(qi[1])][q1b["halfAR"].index(qi[2])] = qi[3]
	# Save results as CSV
	with open("data/q1b.csv", "w", newline="") as f:
		writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(["sweep", "AR", "CL_alpha"])
		for s in range(len(q1b["sweep"])):
			for a in range(len(q1b["AR"])):
				writer.writerow([q1b["sweep"][s], q1b["AR"][a], q1b["CL_alpha"][s][a]])
	# Print results
	run_q1b_cache()
	# Return results
	return q1b

def run_q1b_cache():
	q1b = {"AR":[], "sweep":[], "CL_alpha":[]}
	# Read results from CSV
	with open("data/q1b.csv", "r", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if len(q1b["sweep"]) == 0 or row["sweep"] > q1b["sweep"][-1]:
				q1b["sweep"].append(row["sweep"])
				q1b["CL_alpha"].append([])
				i = -1
			if len(q1b["AR"]) == 0 or row["AR"] > q1b["AR"][-1]:
				q1b["AR"].append(row["AR"])
			i += 1
			q1b["CL_alpha"][-1].append(row["CL_alpha"])
	# Generate plot
	for i in range(len(q1b["sweep"])):
		plt.plot(q1b["AR"], q1b["CL_alpha"][i])
	plt.xlabel("Aspect ratio AR")
	plt.ylabel("CL_alpha")
	plt.title("Effet de l'aspect ratio et de la sweep sur la pente CL_alpha")
	plt.grid()
	plt.savefig("figs/q1b.png") #save plot to file


#----Code Runner----
if __name__ == "__main__":
	run = {"q1a":1, "q1b":1}
	multiprocessing.freeze_support()
	pool = multiprocessing.Pool()
	queue = multiprocessing.Manager().Queue()
	for q in run:
		if run[q] == 1:
			try:
				f = open("data/" + q + ".csv", "r", newline="")
				f.close()
				locals()["run_" + q + "_cache"]()
			except FileNotFoundError:
				locals()["run_" + q](pool, queue)
		elif run[q] == 2:
			locals()["run_" + q](pool, queue)
	pool.terminate()