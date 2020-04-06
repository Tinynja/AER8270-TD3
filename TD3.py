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


def vlm_enqueue(queue, taper_ratio, alphaRange, AR, sweep):
	print(f'Calculating (taper_ratio={taper_ratio}, alphaRange={alphaRange}, AR={AR}, sweep={sweep})...')
	sys.stdout.flush()
	vlm_prob = VLM(ni=5,
				   nj=50,
				   chordRoot=1.0,
				   chordTip=1.0*taper_ratio,
				   twistRoot=0.0,
				   twistTip=0.0,
				   span=AR/2,
				   sweep=sweep,
				   Sref=AR/2,
				   referencePoint=[0.0,0.0,0.0],
				   wingType=1,
				   alphaRange = alphaRange)
	vlm_prob.run()
	queue.put({'taper_ratio':taper_ratio, 'alphaRange':alphaRange, 'AR':AR, 'sweep':sweep, 'prob':vlm_prob})

def full_run(pool, queue, taper_ratio, alphaRange, AR, sweep):
	if not hasattr(taper_ratio, '__iter__'): taper_ratio = (taper_ratio,)
	if not hasattr(alphaRange, '__iter__'): alphaRange = ((alphaRange,),)
	else: alphaRange = [a if hasattr(a, '__iter__') else (a,) for a in alphaRange]
	if not hasattr(AR, '__iter__'): AR = (AR,)
	if not hasattr(sweep, '__iter__'): sweep = (sweep,)
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	vlm_partial = partial(vlm_enqueue, queue)
	pool.starmap(vlm_partial, product(taper_ratio, alphaRange, AR, sweep))
	# Dequeue results
	full = []
	for i in range(queue.qsize()):
		qi = queue.get()
		full.append({
			'taper_ratio': qi['taper_ratio'],
			'alphaRange': qi['alphaRange'],
			'AR': qi['AR'],
			'sweep': qi['sweep'],
			'prob': qi['prob']
		})
	# Return results
	return full

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
	with open('data/q1a.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['AR', 'CL_alpha', 'CD'])
		writer.writerow([q1a.AR, q1a.CL_alpha, q1a.CD[-1]])
	# Return results
	return q1a

def show_q1a():
	q1a = []
	# Read results from CSV
	with open('data/q1a.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			q1a = row
	# Print results
	print('Aspect ratio: %g' % q1a['AR'])
	print('Pente de portance-incidence: %f*pi' % (q1a['CL_alpha']/pi))
	print('Trainee: %e' % q1a['CD'])


#----Question 1 (b)----
def run_q1b(pool, queue):
	# Defining the aspect ratio and CL_alpha lists
	sweep = (0, 30, 45, 60)
	AR = [1e-5] + list(np.arange(0.25,8,0.25))
	alpharange = [0, 10]
	taper_ratio = 1
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	vlm_partial = partial(vlm_enqueue, queue, taper_ratio, alpharange)
	pool.starmap(vlm_partial, product(AR, sweep))
	# Dequeue results
	q1b = {}
	for i in range(queue.qsize()):
		qi = queue.get()
		if qi['sweep'] not in q1b: q1b[qi['sweep']] = {}
		if 'AR' not in q1b[qi['sweep']]: q1b[qi['sweep']]['AR'] = []
		if 'CL_alpha' not in q1b[qi['sweep']]: q1b[qi['sweep']]['CL_alpha'] = []
		q1b[qi['sweep']]['AR'].append(qi['AR'])
		q1b[qi['sweep']]['CL_alpha'].append(qi['prob'].CL_alpha)
	# Save results as CSV
	with open('data/q1b.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['sweep', 'AR', 'CL_alpha'])
		for s in q1b:
			for i in range(len(q1b[s]['AR'])):
				writer.writerow([s, q1b[s]['AR'][i], q1b[s]['CL_alpha'][i]])
	# Return results
	return q1b

def show_q1b():
	# Read results from CSV
	q1b = {}
	with open('data/q1b.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['sweep'] not in q1b: q1b[row['sweep']] = {'AR':[], 'CL_alpha':[]}
			q1b[row['sweep']]['AR'].append(row['AR'])
			q1b[row['sweep']]['CL_alpha'].append(row['CL_alpha'])
	# Read reference values from CSV
	q1b_ref = {}
	with open('data/q1b_ref.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['sweep'] not in q1b_ref: q1b_ref[row['sweep']] = {'AR':[], 'CL_alpha':[]}
			q1b_ref[row['sweep']]['AR'].append(row['AR'])
			q1b_ref[row['sweep']]['CL_alpha'].append(row['CL_alpha'])
	# Generate plot
	plt.figure()
	for s in sorted(q1b):
		plt.plot(q1b[s]['AR'], q1b[s]['CL_alpha'], label=u'\u039B = %d\u00B0' % (s))
	for s in sorted(q1b_ref):
		plt.plot(q1b_ref[s]['AR'], q1b_ref[s]['CL_alpha'], '--', label=u'\u039B %d\u00B0 (ref)' % (s))
	plt.xlabel('Aspect ratio AR')
	plt.ylabel('CL_alpha')
	plt.title('Effet de l\'aspect ratio et de la sweep sur la pente CL_alpha')
	plt.grid()
	plt.legend()
	plt.savefig('figs/q1b.png') #save plot to file


#----Question 1 (c)----
def run_q1c(pool, queue):
	sweep = (7,)
	AR =(7.4,)
	alpha = (10,)
	taper_ratio = (1,)
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	vlm_partial = partial(vlm_enqueue, queue)
	pool.starmap(vlm_partial, product(taper_ratio, alpha, AR, sweep))
	# Dequeue results
	q1c = {}
	q1c['semispan'] =  AR[0]/2
	for i in range(queue.qsize()):
		qi = queue.get()
		q1c[qi['AR']] = qi['prob'].spanLoad[alpha[0]]
		q1c[qi['AR']]['CL'] =  qi['prob'].CL[0]
	# Save results as CSV
	with open('data/q1c.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['AR', '2y/b', 'cl/CL'])
		for s in q1c:
			if not isinstance(s, str):
				for i in range(len(q1c[s]['y'])):
					writer.writerow([s, q1c[s]['y'][i]/q1c['semispan'], q1c[s]['cl_sec'][i]/q1c[s]['CL']])
	# Return results
	return q1c

def show_q1c():
	# Read results from CSV
	q1c = {}
	with open('data/q1c.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['AR'] not in q1c: q1c[row['AR']] = {'2y/b':[], 'cl/CL':[]}
			q1c[row['AR']]['2y/b'].append(row['2y/b'])
			q1c[row['AR']]['cl/CL'].append(row['cl/CL'])
	# Read reference values from CSV
	q1c_ref = {}
	with open('data/q1c_ref.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['sweep'] not in q1c_ref: q1c_ref[row['sweep']] = {'2y/b':[], 'cl/CL':[]}
			q1c_ref[row['sweep']]['2y/b'].append(row['2y/b'])
			q1c_ref[row['sweep']]['cl/CL'].append(row['cl/CL'])
	# Generate plot
	plt.figure()
	for s in sorted(q1c):
		plt.plot(q1c[s]['2y/b'], q1c[s]['cl/CL'], label=u'\u039B = %f\u00B0' % (s))
	for s in sorted(q1c_ref):
		plt.plot(q1c_ref[s]['2y/b'], q1c_ref[s]['cl/CL'], '--', label=u'\u039B = %d\u00B0 (ref)' % (s))
	plt.xlabel('Pourcentage de demi-envergure (2y/b)')
	plt.ylabel('cl/CL')
	plt.title('Effet du sweep sur la distribution de portance')
	plt.grid()
	plt.legend()
	plt.savefig('figs/q1c.png') #save plot to file


#----Question 1 (d)----
def run_q1d(pool, queue):
	AR = 4
	alpha = 10
	taper_ratio = 1
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	vlm_partial = partial(vlm_enqueue, queue, taper_ratio, alpha, AR)
	pool.map(vlm_partial, sweep)
	# Dequeue results
	q1c = {}
	q1c['semispan'] =  AR/2
	for i in range(queue.qsize()):
		qi = queue.get()
		q1c[qi['sweep']] = qi['prob'].spanLoad[alpha]
		q1c[qi['sweep']]['CL'] =  qi['prob'].CL[0]
	# Save results as CSV
	with open('data/q1c.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['sweep', '2y/b', 'cl/CL'])
		for s in q1c:
			if isinstance(s, (int, float)):
				for i in range(len(q1c[s]['y'])):
					writer.writerow([s, q1c[s]['y'][i]/q1c['semispan'], q1c[s]['cl_sec'][i]/q1c[s]['CL']])
	# Return results
	return q1c

def show_q1d():
	pass


#-------Tests-------
def run_q0(pool,queue):
	taper_ratio = 1
	AR = 7.28
	sweep = range(5,10,1)
	alphaRange = 10
	probs = full_run(pool, queue, taper_ratio, alphaRange, AR, sweep)
	# Save results as CSV
	with open('data/q0.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['taper_ratio', 'AR', 'sweep', 'alpha', '2y/b', 'cl/CL'])
		for p in probs:
			for i,alpha in enumerate(p['alphaRange']):
				params = [p['taper_ratio'], p['AR'], p['sweep'], alpha]
				for j,y in enumerate(p['prob'].spanLoad[alpha]['y']):
					writer.writerow(params + [y/p['prob'].span, p['prob'].spanLoad[alpha]['cl_sec'][j]/p['prob'].CL[i]])
	
def show_q0():
	# Read results from CSV
	probs = []
	with open('data/q0.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if len(probs) == 0 or (probs[-1]['taper_ratio'] != row['taper_ratio'] or probs[-1]['AR'] != row['AR'] or probs[-1]['sweep'] != row['sweep'] or probs[-1]['alpha'] != row['alpha']):
				probs.append({'taper_ratio':row['taper_ratio'], 'AR':row['AR'], 'sweep':row['sweep'], 'alpha':row['alpha'], '2y/b':[], 'cl/CL':[]})
			probs[-1]['2y/b'].append(row['2y/b'])
			probs[-1]['cl/CL'].append(row['cl/CL'])
	# Read reference values from CSV
	ref = {}
	with open('data/q1c_ref.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if row['sweep'] not in ref: ref[row['sweep']] = {'2y/b':[], 'cl/CL':[]}
			ref[row['sweep']]['2y/b'].append(row['2y/b'])
			ref[row['sweep']]['cl/CL'].append(row['cl/CL'])
	# Generate plot
	plt.figure()
	for p in probs:
		plt.plot(p['2y/b'], p['cl/CL'], label=u'\u03BB=%.2f, AR=%.2f\u039B, \u039B=%.2f, \u03B1=%.1f' % (p['taper_ratio'], p['AR'], p['sweep'], p['alpha']))
	for s in ref:
		plt.plot(ref[s]['2y/b'], ref[s]['cl/CL'], '--', label='REF')
	plt.xlabel('Pourcentage de demi-envergure (2y/b)')
	plt.ylabel('cl/CL')
	plt.title('Effet du sweep sur la distribution de portance')
	plt.grid()
	plt.legend()
	plt.show()

#----Code Runner----
if __name__ == '__main__':
	questions = {'q0':1, 'q1a':0, 'q1b':0, 'q1c':0, 'q1d':0}
	multiprocessing.freeze_support()
	pool = multiprocessing.Pool()
	queue = multiprocessing.Manager().Queue()
	for q in questions:
		if questions[q] > 0:
			if len(q) == 3:
				print('----Question ' + q[1] + ' (' + q[2] + ')----')
			else:
				print('------Question ' + q[1] + '------')
			if questions[q] == 2:
				locals()['run_' + q](pool, queue)
				locals()['show_' + q]()
			else:
				try:
					with open('data/' + q + '.csv', 'r', newline='') as f: pass
					locals()['show_' + q]()
				except FileNotFoundError:
					locals()['run_' + q](pool, queue)
					locals()['show_' + q]()
	pool.terminate()