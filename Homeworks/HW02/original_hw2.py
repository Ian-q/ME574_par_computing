import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit
import math
from time import time

N = 64000
PI = np.pi
TPB = 32

# INSERT DECORATOR HERE (if needed)
def f21(x):
	'''
	Compute the value of the function in Schorghofer Problem 2.1
	Args:
		x: Float input value
	Returns float function value
	'''
	# INSERT CODE HERE
	pass

def sample_f21(xmin, xmax, n):
	'''
	Compute a uniform sampling of the function f
	Args:
		xmin, xmax: float bounds of sampling interval
		n: int number of sample values
	Returns:
		1D numpy array of float sample values
	'''
	print("Running sample: serial version")
	# INSERT CODE HERE
	pass

# INSERT DECORATOR HERE
def f21_dev(x):
	return 3 * PI**4 * x*x + math.log((x-PI)**2)

# INSERT DECORATOR HERE
def sample_f21_kernel(d_f, d_x):
	'''
	Kernel for evaluating values of f_dev
	Args:
		d_f: 1D float device array for storing computed function values
		d_x: 1D float device array of input values
	Returns:
		None (kernels cannot return values)
	'''
	# INSERT CODE HERE
	pass

def sample_f21_parallel(xmin, xmax, n):
	'''
	Parallelized computation of sample values of f_dev
	Args:
		xmin, xmax: float bounds of sampling interval
		n: int number of sample values
	Returns:
		1D numpy array of float sample values
	'''
	print("Running sample_parallel")
	# INSERT CODE HERE
	pass

def p1():
	'''
	Test codes for problem 1
	'''
	xmin, xmax = 0,4
	n = 500000
	x = np.linspace(xmin, xmax, n)
	y = sample_f21(xmin, xmax, n)
	plt.plot(x, y, label='serial')

	y_par = sample_f21_parallel(xmin, xmax, n)
	diff = np.absolute(y - y_par)
	maxDiff = np.max(diff)
	print(maxDiff)
	print("Max. Diff. = ", maxDiff)
	plt.plot(x, diff, label = 'difference')
	plt.plot(x, y_par, label ='parallel')

	plt.legend()
	plt.show()

def time_f21(xmin, xmax, n):
	'''
	Compute a uniform sampling of the function f
	Args:
		xmin, xmax: float bounds of sampling interval
		n: int number of sample values
	Returns:
		Timing for evaluation
	'''
	# INSERT CODE HERE
	pass


def time_f21_parallel(xmin, xmax, n):
	'''
	Parallelized computation of sample values of f_dev
	Args:
		xmin, xmax: float bounds of sampling interval
		n: int number of sample values
	Returns:
		Kernel execution time
	'''
	# INSERT CODE HERE
	pass

def p2():
	'''
	Test codes for problem 2
	'''
	xmin, xmax = 0,4
	n = 500000
	print(f'n = {n:6d}')
	x = np.linspace(xmin, xmax, n)
	t = 1000*time_f21(xmin, xmax, n) # convert to msec
	print(f'Serial time = {t:0.3f} msec')
	t_par = time_f21_parallel(xmin, xmax, n)
	print(f'Parallel time = {t_par:0.3f} msec')
	print(f'Accleration = {t/t_par:0.3f}')

# INSERT DECORATOR HERE
def f(x, a):
	'''
	Compute value of logistic map
	Args:
		x: float input value
		a: float parameter value
	Returns:
		float function value
	'''
	return a*x*(1-x)


def iterate(x, a, k):
	'''
	Compute an array of values for the k^th iteration of the function f(x)
	Args:
		x: numpy array of float input values
		a: float parameter value
		k: int iteration number
	Returns:
		numpy array of k^th iterate values
	'''
	# INSERT CODE HERE
	pass

# INSERT DECORATOR HERE
def f_dev(x, a):
	'''
	Compute value of logistic map
	Args:
		x: float input value
		a: float parameter value
	Returns:
		float function value
	'''
	return a*x*(1-x)

# INSERT DECORATOR HERE
def iterate_kernel(d_f, d_x, a, k):
	# INSERT CODE HERE
	pass 


TPB = 32
def iterate_parallel(x, a, k):
	'''
	Compute an array of values for the k^th iteration of the function f(x)
	Args:
		x: numpy array of float input values
		a: float parameter value
		k: int iteration number
	Returns:
		numpy array of k^th iterate values
	'''
	# INSERT CODE HERE
	pass

def p3():
	'''
	Test codes for problem 3
	'''
	n = 10000
	k = 64
	a = 3.57
	x = np.linspace(0,1,n)
	fk = iterate(x,a,k)
	fk_par = iterate_parallel(x, a, k)
	diff = fk - fk_par
	print("Max. diff. = ", np.max(np.abs(diff)))
	plt.plot(x,x)
	plt.plot(x,fk)
	plt.plot(x,diff)
	plt.show()


def main():
	p1()
	p2()
	p3()

if __name__ == '__main__':
	main()

