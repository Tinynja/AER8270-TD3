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
	# Return results
	return q1a

def show_q1a():
	q1a = []
	# Read results from CSV
	with open("data/q1a.csv", "r", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			q1a = row
	# Print results
	print("Aspect ratio: %g" % q1a["AR"])
	print("Pente de portance-incidence: %f*pi" % (q1a["CL_alpha"]/pi))
	print("Trainee: %e" % q1a["CD"])

#----Question 1 (b)----
def q1b_enqueue(queue, sweep, ar):
	print(f"Calculating (sweep={sweep}, AR={ar})...")
	sys.stdout.flush()
	vlm_prob = VLM(ni=5,
				   nj=50,
				   chordRoot=1.0,
				   chordTip=1.0,
				   twistRoot=0.0,
				   twistTip=0.0,
				   span=ar/2,
				   sweep=sweep,
				   Sref=ar/2,
				   referencePoint=[0.0,0.0,0.0],
				   wingType=1,
				   alphaRange = [0,10])
	vlm_prob.run()
	queue.put({'queue_type':"prob", 'sweep':sweep, 'ar':ar, 'data':vlm_prob})
	queue.put({'queue_type':"CL_alpha", 'sweep':sweep, 'ar':ar, 'data':vlm_prob.CL_alpha})

def run_q1b(pool, queue):
	# Defining the aspect ratio and CL_alpha lists
	sweep = (0, 30, 45, 60)
	AR = [1e-5] + list(np.arange(0.25,8,0.25))
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	vlm_partial = partial(q1b_enqueue, queue)
	pool.starmap(vlm_partial, product(q1b["sweep"], q1b["AR"]))
	# Dequeue results
	q1b = {}
	for i in range(queue.qsize()):
		qi = queue.get()
		if qi['sweep'] not in q1b: q1b[qi['sweep']] = {}
		if 'AR' not in q1b[qi['sweep']]: q1b[qi['sweep']]['AR'] = []
		if qi['queue_type'] not in q1b[qi['sweep']]: q1b[qi['sweep']][qi['queue_type']] = []
		q1b[qi['sweep']]['AR'].append(qi['AR'])
		q1b[qi['sweep']][qi['queue_type']].append(qi['data'])
	# Save results as CSV
	with open("data/q1b.csv", "w", newline="") as f:
		writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(["sweep", "AR", "CL_alpha"])
		for s in q1b:
			for i in range(len(q1b[s]["AR"])):
				writer.writerow(s, q1b[s]['AR'][i], q1b[s]['CL_alpha'][i])
	# Return results
	return q1b

def show_q1b():
	# Read results from CSV
	q1b = {}
	with open("data/q1b.csv", "r", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['sweep'] not in q1b: q1b[row['sweep']] = {}
			if 'AR' not in q1b[row['sweep']]: q1b[row['sweep']]['AR'] = []
			if 'CL_alpha' not in q1b[row['sweep']]: q1b[row['sweep']]['CL_alpha'] = []
			q1b[row['sweep']]['AR'].append(row['AR'])
			q1b[row['sweep']]['CL_alpha'].append(row['CL_alpha'])
	# Read reference values from CSV
	q1b_ref = {}
	with open("data/q1b_ref.csv", "r", newline="") as f:
		reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['sweep'] not in q1b_ref: q1b_ref[row['sweep']] = {}
			if 'AR' not in q1b_ref[row['sweep']]: q1b_ref[row['sweep']]['AR'] = []
			if 'CL_alpha' not in q1b_ref[row['sweep']]: q1b_ref[row['sweep']]['CL_alpha'] = []
			q1b_ref[row['sweep']]['AR'].append(row['AR'])
			q1b_ref[row['sweep']]['CL_alpha'].append(row['CL_alpha'])
	# Generate plot
	plt.figure()
	for s in q1b:
		plt.plot(q1b[s]['AR'], q1b[s]['CL_alpha'], label=u'\u039B = %d\u00B0' % (s))
	for s in q1b_ref:
		plt.plot(q1b_ref[s]['AR'], q1b_ref[s]['CL_alpha'], '--', label=u'\u039B %d\u00B0 (ref)' % (s))
	plt.xlabel("Aspect ratio AR")
	plt.ylabel("CL_alpha")
	plt.title("Effet de l'aspect ratio et de la sweep sur la pente CL_alpha")
	plt.grid()
	plt.legend()
	plt.savefig("figs/q1b.png") #save plot to file

#----Question 1 (c)----

#----Code Runner----
if __name__ == "__main__":
	questions = {"q1a":1, "q1b":1}
	multiprocessing.freeze_support()
	pool = multiprocessing.Pool()
	queue = multiprocessing.Manager().Queue()
	for q in questions:
		if questions[q] > 0:
			if len(q) == 3:
				print("----Question " + q[1] + " (" + q[2] + ")----")
			else:
				print("------Question " + q[1] + "------")
			if questions[q] == 2:
				locals()["run_" + q](pool, queue)
				locals()["show_" + q]()
			else:
				try:
					with open("data/" + q + ".csv", "r", newline="") as f: pass
					locals()["show_" + q]()
				except FileNotFoundError:
					locals()["run_" + q](pool, queue)
					locals()["show_" + q]()
	pool.terminate()