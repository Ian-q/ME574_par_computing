# ME574 Spr2025
# Homework 3
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit, float64
from math import sin, sinh
from time import time

N = 6400000
PI = np.pi
TPB = 64
RAD = 3
NSHARED = 70 #TPB + 2*RAD = 64 + 2*3 = 70
MSPERSEC = 1000
BLOCKSPERSM = 4

@jit #decorate to compile for both host and device
def s(x0):
	'''
	Compute the value of the function sin(PI*x)
	Args:
		x0: float input value
	Returns float function value
	'''
	return sin(PI*x0)

@jit #decorate to compile for both host and device
def u(x0, y0):
	'''
	Compute the value of the function sin(PI*x0)*sinh(PI*y0)
	Args:
		x0: float input value
		y0: float input value
	Returns float function value
	'''
	return sin(PI*x0)*sinh(PI*y0)


def timed_serial(x):
	'''
	Compute array of values of the function sin(PI*x)
	Args:
		x: numpy float array of input values
	Returns:
		float evaluation time in ms	
		float array of function values
	'''
	# INSERT CODE HERE
	pass

@cuda.jit('void(f8[:], f8[:])')
def mono_kernel(d_out, d_x):
	'''
	Monolithic kernel to compute arrays of s(x)
	Args:
		d_out: float device array to store output
		d_x: float device input array
	Returns: None
	'''
	# INSERT CODE HERE
	pass


def timed_parallel(x):
	'''
	Compute array of values of the function sin(PI*x)
	Args:
		x: numpy float array of input values
	Returns:
		float evaluation time in ms
		float data transfer time in ms	
		float array of function values
	'''
	# INSERT CODE HERE
	pass

@cuda.jit('void(f8[:], f8[:])')
def gridstride_kernel(d_out, d_x):
	'''
	Monolithic kernel to compute arrays of s(x)
	Args:
		d_out: float device array to store output
		d_x: float device input array
	Returns: None
	'''
	# INSERT CODE HERE
	pass

def timed_gridstride(x):
	'''
	Compute array of values of the function sin(PI*x) using gridstride loop
	Args:
		x: numpy float array of input values
	Returns:
		float evaluation time in ms
		float data transfer time in ms	
		float array of function values
	'''
	# INSERT CODE HERE
	pass



def p1():
	'''
	Test codes for problem 1
	'''
	print('Problem 1:')
	n = 500
	x = np.linspace(0, 1, n)
	t_ser, v_ser = timed_serial(x)
	t_mono, t_monocopy, v_mono = timed_parallel(x)
	t_gs, t_gscopy, v_gs = timed_gridstride(x)
	diff = np.absolute(v_ser - v_mono)
	maxDiff = np.max(diff)
	print(maxDiff)
	print("Max. Diff. = ", maxDiff)
	print('t_ser: ', t_ser)
	print('t_mono: ', t_mono + t_monocopy)
	print('t_gs: ', t_gs)
	print('t_copy: ', t_monocopy)
	print(f'Mono Acceleration: {t_ser/(t_mono + t_monocopy):0.2f}')
	print(f'Added gridstride acceleration: {(t_mono + t_monocopy)/(t_gs + t_gscopy):0.2f}')
	plt.plot(x, diff, label = 'difference')
	plt.plot(x, v_mono, 'x', label ='parallel')
	plt.plot(x, v_gs, label ='gridstride')
	plt.legend()
	plt.show()

@cuda.jit('void(f8[:], f8[:], f8[:])')
def ode_kernel(d_out, d_s, d_stencil):
	'''
	Kernel to compute arrays of d2s/dx2 + s using global memory
	Args:
		d_out: float device array to store output
		d_s: float device input array
		d_stencil: float device array of stencil coefficients
	Returns: None
	'''
	# INSERT CODE HERE
	pass


def ode_check(s, h):
	'''
	Compute array of values for d2s/dx2 + s
	Args:
		s: numpy float array of input values
		h: float sample spacing
	Returns:
		float evaluation time in ms
		float data transfer time in ms	
		float array of function values
	'''
	# INSERT CODE HERE
	pass

################# shared memory versions ##########

@cuda.jit('void(f8[:], f8[:], f8[:])')
def sh_ode_kernel(d_out, d_s, d_stencil):
	'''
	Kernel to compute arrays of d2s/dx2 + s using global memory
	Args:
		d_out: float device array to store output
		d_s: float device input array
		d_stencil: float device array of stencil coefficients
	Returns: None
	'''
	# INSERT CODE HERE
	pass


def sh_ode_check(s, h):
	'''
	Compute array of values for d2s/dx2 + s
	Args:
		s: numpy float array of input values
		h: float sample spacing
	Returns:
		float evaluation time in ms
		float data transfer time in ms	
		float array of function values
	'''
	# INSERT CODE HERE
	pass

def p2(verbose=False):
	'''
	Test codes for problem 2
	'''
	print('Problem 2:')
	n = 50000
	x = np.linspace(0, 1, n, dtype=float)
	h = x[1]
	t1,t2,s = timed_gridstride(x)
	if verbose:
		plt.plot(x,s)
		plt.show()
	t_e, t_c, g_mem = ode_check(s, h)
	print('Evaluation time (global)= ', t_e)
	print('Max error(global): ', np.max(np.abs(g_mem)))
	if verbose:	
		plt.plot(x, g_mem)
		plt.ylim(-2,2)
		plt.show()
	t_e, t_c, sh_mem = sh_ode_check(s, h)
	print('Evaluation time (shared)= ', t_e)
	print('Max error (shared): ', np.max(np.abs(sh_mem)))
	if verbose:
		plt.plot(x, sh_mem)
		plt.ylim(-2,2)
		plt.show()
	# shared = sh_ode_check(x)



def main():
	p1()
	p2()

if __name__ == '__main__':
	main()

