import multiprocessing as mp
from functools import partial
from itertools import product

def f(q,y):
	q.put(y)

class A(object):
	def __init__(self, njobs=1000):
		self.njobs = njobs
	def start(self):
		self.p = mp.Pool()
		self.result = self.p.map(self.RunProcess, range(self.njobs))
	def RunProcess(self, i):
		return i*i

if __name__ == '__main__':
	mp.freeze_support()
	q = mp.Manager().Queue()
	y = [0, 1, 2]
	z = [4, 5, 6]
	func = partial(f, q)
	p = mp.Pool()
	p.map(func, y)
	p.terminate()