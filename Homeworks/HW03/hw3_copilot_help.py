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
	time_start = time()
	y = np.array([s(x_i) for x_i in x])
	time_stop = time()
	duration = (time_stop - time_start) * 1000  # time in ms
	
	return(duration, y)

	time_start = time()
	y = np.array([s(x_i) for x_i in x])
	time_stop = time()
	duration = (time_stop - time_start) * 1000  # time in ms
	
	return(duration, y)


@cuda.jit('void(f8[:], f8[:])')
def mono_kernel(d_out, d_x):
	'''
	Monolithic kernel to compute arrays of s(x)
	Args:
		d_out: float device array to store output
		d_x: float device input array
	Returns: None
	'''
	idx = cuda.grid(1)
	if idx < d_x.shape[0]:
		d_out[idx] = s(d_x[idx])


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
	n = x.shape[0]
	d_x = cuda.to_device(x)
	d_out = cuda.device_array_like(d_x)
	
	threadsperblock = TPB
	blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
	
	start_event = cuda.event()
	end_event = cuda.event()
	
	start_event.record()
	mono_kernel[blockspergrid, threadsperblock](d_out, d_x)
	end_event.record()
	end_event.synchronize()
	eval_time = cuda.event_elapsed_time(start_event, end_event) # time in ms

	start_copy_event = cuda.event()
	end_copy_event = cuda.event()

	start_copy_event.record()
	y = d_out.copy_to_host()
	end_copy_event.record()
	end_copy_event.synchronize()
	copy_time = cuda.event_elapsed_time(start_copy_event, end_copy_event) # time in ms

	return eval_time, copy_time, y

@cuda.jit('void(f8[:], f8[:])')
def gridstride_kernel(d_out, d_x):
	'''
	Gridstride kernel to compute arrays of s(x)
	Args:
		d_out: float device array to store output
		d_x: float device input array
	Returns: None
	'''
	start = cuda.grid(1)
	stride = cuda.gridsize(1)
	for i in range(start, d_x.shape[0], stride):
		d_out[i] = s(d_x[i])

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
	n = x.shape[0]
	d_x = cuda.to_device(x)
	d_out = cuda.device_array_like(d_x)
	
	device = cuda.get_current_device()
	num_sm = device.MULTIPROCESSOR_COUNT
	threadsperblock = TPB
	blockspergrid = num_sm * BLOCKSPERSM
	
	start_event = cuda.event()
	end_event = cuda.event()
	
	start_event.record()
	gridstride_kernel[blockspergrid, threadsperblock](d_out, d_x)
	end_event.record()
	end_event.synchronize()
	eval_time = cuda.event_elapsed_time(start_event, end_event) # time in ms

	start_copy_event = cuda.event()
	end_copy_event = cuda.event()

	start_copy_event.record()
	y = d_out.copy_to_host()
	end_copy_event.record()
	end_copy_event.synchronize()
	copy_time = cuda.event_elapsed_time(start_copy_event, end_copy_event) # time in ms

	return eval_time, copy_time, y


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

@cuda.jit('void(f8[:], f8[:], f8[:], f8)')
def ode_kernel(d_out, d_s, d_stencil, h):
    '''
    Kernel to compute arrays of d2s/dx2 + pi^2*s using global memory
    Args:
        d_out: float device array to store output
        d_s: float device input array
        d_stencil: float device array of stencil coefficients
        h: float sample spacing
    Returns: None
    '''
    idx = cuda.grid(1)
    n = d_s.shape[0]
    
    # Handle boundaries
    if idx < RAD or idx >= n - RAD:
        d_out[idx] = 0.0
        return

    # Compute second derivative using stencil
    d2s_dx2 = 0.0
    for i in range(len(d_stencil)):
        d2s_dx2 += d_stencil[i] * d_s[idx + i - RAD]
    
    d2s_dx2 /= (180.0 * h**2)
    
    # Compute the ODE check value
    d_out[idx] = d2s_dx2 + (PI**2) * d_s[idx]

def ode_check(s_vals, h):
    '''
    Compute array of values for d2s/dx2 + pi^2*s
    Args:
        s_vals: numpy float array of input function values
        h: float sample spacing
    Returns:
        float evaluation time in ms
        float data transfer time in ms (not required by description but good practice)
        float array of function values
    '''
    n = s_vals.shape[0]
    stencil = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    
    d_s = cuda.to_device(s_vals)
    d_stencil = cuda.to_device(stencil)
    d_out = cuda.device_array_like(d_s)
    
    threadsperblock = TPB
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
    
    start_event = cuda.event()
    end_event = cuda.event()
    
    start_event.record()
    ode_kernel[blockspergrid, threadsperblock](d_out, d_s, d_stencil, h)
    end_event.record()
    end_event.synchronize()
    eval_time = cuda.event_elapsed_time(start_event, end_event) # time in ms

    start_copy_event = cuda.event()
    end_copy_event = cuda.event()

    start_copy_event.record()
    ode_result = d_out.copy_to_host()
    end_copy_event.record()
    end_copy_event.synchronize()
    copy_time = cuda.event_elapsed_time(start_copy_event, end_copy_event) # time in ms

    # The description asks for a 2-tuple (eval_time, result_array)
    return eval_time, ode_result

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

