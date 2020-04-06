import multiprocessing
from functools import partial

def add(a,b):
	return a+b

if __name__ == "__main__":
	multiprocessing.freeze_support()
	pool = multiprocessing.Pool()
	queue = multiprocessing.Manager().Queue()
	numbers = (0,2,3,5,7,9)
	partial_add = partial(add, 2)
	print(pool.map(partial_add, numbers))
	pool.terminate()