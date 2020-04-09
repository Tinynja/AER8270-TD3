# Standard libs
from math import *
import multiprocessing
import csv
import re
from os import listdir
from functools import partial
from itertools import product

# Pip libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# User libs
from libraries.vlm import *


def vlm_enqueue(queue, taper_ratio, alphaRange, sweep, AR, span=None, twist=(0,0), wingtype=1):
	if span is not None:
		if wingtype in (1,2):
			AR = 4*span/(1+taper_ratio)
		elif wingtype == 3:
			AR = 8*span/pi
	if wingtype in (1,2):
		span = AR*(1+taper_ratio)/4
		Sref = AR*(1+taper_ratio)**2/8
	elif wingtype == 3:
		span = AR*pi/8
		Sref = AR*pi**2/32
	print(f'Calculating (taper_ratio={taper_ratio}, alphaRange={alphaRange}, sweep={sweep}, AR={AR:g}, span={span}, twist={twist}, wingtype={wingtype})...', flush=True)
	vlm_prob = VLM(ni=5,
				   nj=50,
				   chordRoot=1.0,
				   chordTip=1.0*taper_ratio,
				   twistRoot=twist[0],
				   twistTip=twist[1],
				   span=span,
				   sweep=sweep,
				   Sref=Sref,
				   referencePoint=[0.0,0.0,0.0],
				   wingType=wingtype,
				   alphaRange = alphaRange)
	vlm_prob.run()
	queue.put({'taper_ratio':taper_ratio, 'alphaRange':alphaRange, 'sweep':sweep, 'AR':AR,
						 'span':span, 'twist':twist, 'wingtype':wingtype, 'prob':vlm_prob})

def full_run(pool, queue, taper_ratio, alphaRange, sweep, AR, span=None, twist=(0,0), wingtype=1, asis=False):
	if not hasattr(taper_ratio, '__iter__'): taper_ratio = (taper_ratio,)
	if not hasattr(alphaRange, '__iter__'): alphaRange = ((alphaRange,),)
	else: alphaRange = [a if hasattr(a, '__iter__') else (a,) for a in alphaRange]
	if not hasattr(sweep, '__iter__'): sweep = (sweep,)
	if not hasattr(AR, '__iter__'): AR = (AR,)
	if not hasattr(span, '__iter__'): span = (span,)
	assert hasattr(twist, '__iter__'), 'twist argument must be a list: (twistRoot, twistTip)'
	if not hasattr(twist[0], '__iter__'):
		twist = (twist[0:2],)
	if not hasattr(wingtype, '__iter__'): wingtype = (wingtype,)
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	params = []
	if asis:
		args = ['taper_ratio', 'alphaRange', 'sweep', 'AR', 'span', 'twist', 'wingtype']
		new_args = []
		locals_ = locals()
		repl = max([len(locals_[a]) for a in args])
		for a in args:
			l = len(locals_[a])
			assert l in (1, repl), f'Length of argument `{a}` should be 1 or {repl}.'
			if l == 1:
				new_args.append(locals()[a]*repl)
			else:
				new_args.append(locals()[a])
		params = zip(*new_args)
	else:
		params = product(taper_ratio, alphaRange, sweep, AR, span, twist, wingtype)
	vlm_partial = partial(vlm_enqueue, queue)
	pool.starmap(vlm_partial, params)
	# Dequeue results
	full = []
	for i in range(queue.qsize()):
		qi = queue.get()
		full.append(qi)
	# Return results
	return full

def save_data(results, filename, datatype):
	with open(f'data/{filename}.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		row = ['taper_ratio', 'alphaRange', 'sweep', 'AR']
		if datatype == 'alpha-CL':
			writer.writerow(row + ['CL',])
		elif datatype == 'AR-CL_alpha':
			writer.writerow(row + ['CL_alpha',])
		elif datatype == 'y/b-cl/CL':
			writer.writerow(row + ['y/b', 'cl/CL'])
		elif datatype == 'y/b-clclocal/cavg':
			writer.writerow(row + ['y/b', 'clclocal/cavg'])
		for r in results:
			if datatype == 'alpha-CL':
				for i,alpha in enumerate(r['alphaRange']):
					writer.writerow([r['taper_ratio'], alpha, r['sweep'], r['AR'], r['prob'].CL[i]])
			elif datatype == 'AR-CL_alpha':
				writer.writerow([r['taper_ratio'], r['alphaRange'], r['sweep'], r['AR'], r['prob'].CL_alpha,])
			elif datatype == 'y/b-cl/CL':
				params = [r['taper_ratio'], r['alphaRange'][-1], r['sweep'], r['AR']]
				for j,y in enumerate(r['prob'].spanLoad[-1]['y']):
					writer.writerow(params + [y/r['prob'].span, r['prob'].spanLoad[-1]['cl_sec'][j]/r['prob'].CL[-1]])
			elif datatype == 'y/b-clclocal/cavg':
				params = [r['taper_ratio'], r['alphaRange'][-1], r['sweep'], r['AR']]
				for j,y in enumerate(r['prob'].spanLoad[-1]['y']):
					writer.writerow(params + [y/r['prob'].span, r['prob'].spanLoad[-1]['cl_sec'][j]*r['prob'].clocal[j]/r['prob'].cavg])

def read_data(filename, datatype, sort_by=None):
	# Reads results from CSV file and saves them as (depending on datatype):
	# results = [{'taper_ratio':__, 'sweep':__, 'AR':__, 'alphaRange':[__, __, __], 'CL':[__, __, __]}, {...}, ...]
	# Read results from CSV
	results = []
	with open(f'data/{filename}.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			if datatype == 'alpha-CL':
				if len(results) == 0 or (results[-1]['taper_ratio'] != row['taper_ratio'] or results[-1]['sweep'] != row['sweep'] or results[-1]['AR'] != row['AR']):
					results.append({'taper_ratio':row['taper_ratio'], 'AR':row['AR'], 'sweep':row['sweep'], 'alphaRange':[], 'CL':[]})
				results[-1]['alphaRange'].append(row['alphaRange'])
				results[-1]['CL'].append(row['CL'])
			elif datatype == 'AR-CL_alpha':
				if len(results) == 0 or (results[-1]['taper_ratio'] != row['taper_ratio'] or results[-1]['alphaRange'] != row['alphaRange'] or results[-1]['sweep'] != row['sweep']):
					results.append({'taper_ratio':row['taper_ratio'], 'alphaRange':row['alphaRange'], 'sweep':row['sweep'], 'AR':[], 'CL_alpha':[]})
				results[-1]['AR'].append(row['AR'])
				results[-1]['CL_alpha'].append(row['CL_alpha'])
			elif datatype == 'y/b-cl/CL':
				if len(results) == 0 or (results[-1]['taper_ratio'] != row['taper_ratio'] or results[-1]['alphaRange'] != row['alphaRange'] or results[-1]['sweep'] != row['sweep'] or results[-1]['AR'] != row['AR']):
					results.append({'taper_ratio':row['taper_ratio'], 'alphaRange':row['alphaRange'], 'sweep':row['sweep'], 'AR':row['AR'], 'y/b':[], 'cl/CL':[]})
				results[-1]['y/b'].append(row['y/b'])
				results[-1]['cl/CL'].append(row['cl/CL'])
			elif datatype == 'y/b-clclocal/cavg':
				if len(results) == 0 or (results[-1]['taper_ratio'] != row['taper_ratio'] or results[-1]['alphaRange'] != row['alphaRange'] or results[-1]['sweep'] != row['sweep'] or results[-1]['AR'] != row['AR']):
					results.append({'taper_ratio':row['taper_ratio'], 'alphaRange':row['alphaRange'], 'sweep':row['sweep'], 'AR':row['AR'], 'y/b':[], 'clclocal/cavg':[]})
				results[-1]['y/b'].append(row['y/b'])
				results[-1]['clclocal/cavg'].append(row['clclocal/cavg'])
	if sort_by is not None:
		return sorted(results, key=lambda k: k[sort_by])
	return results


#----Question 1 (a)----
def run_q1a(pool, queue):
	# Run VLM
	result = full_run(pool, queue, 1, [[0, 10],], 0, 1e10)[0]['prob']
	# Save results as CSV
	with open('data/q1a.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['AR', 'CL_alpha', 'CD'])
		writer.writerow([result.AR, result.CL_alpha, result.CD[-1]])
	# Return result
	return result

def show_q1a():
	result = []
	# Read results from CSV
	with open('data/q1a.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			result = row
	# Print results
	print('Aspect ratio: %g' % result['AR'])
	print('Pente de portance-incidence: %f*pi' % (result['CL_alpha']/pi))
	print('Trainee: %e' % result['CD'])


#----Question 1 (b)----
def run_q1b(pool, queue):
	# Defining the aspect ratio and CL_alpha lists
	taper_ratio = 1
	alpharange = [[0,10]]
	sweep = (0, 30, 45, 60)
	AR = [1e-5] + list(np.arange(0.25,8,0.25))
	# Enqueuing VLM results from multiprocessor pool for each combination of AR and sweep
	results = full_run(pool, queue, taper_ratio, alpharange, sweep, AR)
	# Save results as CSV
	save_data(results, 'q1b', 'AR-CL_alpha')
	# Return result
	return results

def show_q1b():
	# Read results from CSV
	results = read_data('q1b', 'AR-CL_alpha', sort_by='sweep')
	# Read reference values from CSV
	results_ref = read_data('q1b_ref', 'AR-CL_alpha', sort_by='sweep')
	# Generate plot
	plt.figure()
	for r in results:
		plt.plot(r['AR'], r['CL_alpha'], label=u'\u039B = %d\u00B0' % (r['sweep']))
	for r in results_ref:
		plt.plot(r['AR'], r['CL_alpha'], '--', label=u'\u039B %d\u00B0 (ref)' % (r['sweep']))
	plt.xlabel('Aspect ratio AR')
	plt.ylabel('CL_alpha')
	plt.title('Effet de l\'aspect ratio et de la sweep sur la pente CL_alpha')
	plt.grid()
	plt.legend()
	plt.savefig('figs/q1b.png') #save plot to file


#----Question 1 (c)----
def run_q1c(pool, queue):
	taper_ratio = 1
	alpha = 10
	sweep = (0, 45, 135)
	AR = 4
	# Running the VLMs
	results = full_run(pool, queue, taper_ratio, alpha, sweep, AR)
	# Save results as CSV
	save_data(results, 'q1c', 'y/b-cl/CL')
	# Return results
	return results

def show_q1c():
	# Read results from CSV
	results = read_data('q1c', 'y/b-cl/CL', sort_by='sweep')
	# Read reference values from CSV
	results_ref = read_data('q1c_ref', 'y/b-cl/CL', sort_by='sweep')
	# Generate plot
	plt.figure()
	for r in results:
		plt.plot(r['y/b'], r['cl/CL'], label=u'\u039B = %f\u00B0' % (r['sweep']))
	for r in results_ref:
		plt.plot(r['y/b'], r['cl/CL'], '--', label=u'\u039B = %d\u00B0 (ref)' % (r['sweep']))
	plt.xlabel('Pourcentage de demi-envergure (2y/b)')
	plt.ylabel('cl/CL')
	plt.title('Effet du sweep sur la distribution de portance')
	plt.grid()
	plt.legend()
	plt.savefig('figs/q1c.png') #save plot to file


#----Question 1 (d)----
def run_q1d(pool, queue):
	taper_ratio = (0, 0.4, 0.6, 1)
	alpha = 10
	sweep = 0
	AR = 7.28
	# Running the VLMs
	results = full_run(pool, queue, taper_ratio, alpha, sweep, AR)
	# Save results as CSV
	save_data(results, 'q1d', 'y/b-cl/CL')
	# Return results
	return results

def show_q1d():
	# Read results from CSV
	results = read_data('q1d', 'y/b-cl/CL', sort_by='taper_ratio')
	# Read reference values from CSV
	results_ref = read_data('q1d_ref', 'y/b-cl/CL', sort_by='taper_ratio')
	# Generate plot
	plt.figure()
	for r in results:
		plt.plot(r['y/b'], r['cl/CL'], label=u'\u03BB = %.2f' % (r['taper_ratio']))
	for r in results_ref:
		plt.plot(r['y/b'], r['cl/CL'], '--', label=u'\u03BB = %.2f (ref)' % (r['taper_ratio']))
	plt.ylim(0, 1.5)
	plt.xlabel('Pourcentage de demi-envergure (2y/b)')
	plt.ylabel('cl/CL')
	plt.title('Effet du taper ratio sur la distribution de portance')
	plt.grid()
	plt.legend()
	plt.savefig('figs/q1d.png') #save plot to file


#----Question 1 (e)----
def run_q1e(pool, queue):
	taper_ratio = 1
	alpha = ((0,10),)
	sweep = 0
	AR = 0
	span = 20
	# Run VLM
	result = full_run(pool, queue, taper_ratio, alpha, sweep, AR, span, wingtype=3)
	# Save results as CSV
	with open('data/q1e-i.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['AR', 'CL', 'CD', 'CL_alpha'])
		writer.writerow([result[0]['AR'], result[0]['prob'].CL[-1], result[0]['prob'].CD[-1], result[0]['prob'].CL_alpha])
	save_data(result, 'q1e-ii', 'y/b-cl/CL')
	# Return result
	return result

def show_q1e():
	# Read results from CSV
	with open('data/q1e-i.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			result_i = row
	# Read results from CSV
	result_ii = read_data('q1e-ii', 'y/b-cl/CL')
	# Print results
	print('Aspect ratio (AR): %g' % result_i['AR'])
	print('Pente de portance-incidence: %f*pi/(1+2/AR)' % (result_i['CL_alpha']*(1+2/result_i['AR'])/pi))
	print('Coefficient d\'Oswald: %f' % (result_i['CL']**2/(pi*result_i['CD']*result_i['AR'])))
	#e = CL[1]**2 / (math.pi * CD[1] * AR)
	# Generate plot
	plt.figure()
	for r in result_ii:
		plt.plot(r['y/b'], r['cl/CL'])
	plt.xlabel('Pourcentage de demi-envergure (2y/b)')
	plt.ylabel('cl/CL')
	plt.title('Distribution de portante d\'une aile elliptique')
	plt.grid()
	plt.savefig('figs/q1e-ii.png') #save plot to file


#------Question 3------
def run_q3(pool, queue):
	taper_ratio = 0.4
	alphaRange = (-10, 10)
	sweep = 35
	AR = 0
	span = 10
	twist = ((0,-4), (0,0), (0,4))
	results = full_run(pool, queue, taper_ratio, alphaRange, sweep, AR, span, twist)
	CL = [[], [], []]
	for r in results:
		CL[[t[-1] for t in twist].index(r['twist'][-1])].append(r['prob'].CL[0])
	newalphaRange = []
	for i,CLi in enumerate(CL):
		newalphaRange.append(interp1d(sorted(CLi), alphaRange)(0.5)+0)
	results = full_run(pool, queue, taper_ratio, newalphaRange, sweep, AR, span, twist, asis=True)
	# Save results as CSV
	with open('data/q3-i.csv', 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['twist', 'CDi'])
		for r in results:
			writer.writerow([r['twist'][-1], r['prob'].CDi[0]])
	save_data(results, 'q3-ii', 'y/b-clclocal/cavg')
	# Return results
	return results

def show_q3():
	# Read results from CSV
	with open('data/q3-i.csv', 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
		for row in reader:
			results_i = row
	# Read results from CSV
	results_ii = read_data('q3-ii', 'y/b-clclocal/cavg')
	# Print results
	for r in results_i:
		print(f"Coefficient de trainée induite (twist = {r['twist']}): {r['CDi']}")
	# Generate plot
	plt.figure()
	for r in results_ii:
		plt.plot(r['y/b'], r['y/b-clclocal/cavg'], label=u'Twist=%s\u00B0' % (r['twist'][-1]))
	plt.xlabel('Pourcentage de demi-envergure (2y/b)')
	plt.ylabel('cl/CL')
	plt.title('Distribution de portante d\'une aile elliptique')
	plt.grid()
	plt.legend()
	plt.savefig('figs/q3-ii.png') #save plot to file


#-------Tests-------
def run_q0(pool,queue):
	taper_ratio = 1
	alphaRange = 10
	sweep = 0
	AR = 0
	span = 10
	# Running the VLMs
	results = full_run(pool, queue, taper_ratio, alphaRange, sweep, AR, twist=(0,10), span=span)
	
def show_q0():
	pass


#----Code Runner----
if __name__ == '__main__':
	questions = {'q0':0, 'q1a':0, 'q1b':0, 'q1c':0, 'q1d':0, 'q1e':0, 'q3':2}
	datafiles = listdir('data')
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
				if any(re.match(f'{q}-?i*\.csv',f_name) for f_name in datafiles):
					locals()['show_' + q]()
				else:
					locals()['run_' + q](pool, queue)
					locals()['show_' + q]()
	pool.terminate()