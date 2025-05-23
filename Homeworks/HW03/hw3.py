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
	
	# ensure that idx is within array bounds
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
	# copy input array to gpu memory
	d_x = cuda.to_device(x)
	# initialize output array for kernel function
	d_out = cuda.device_array(x.shape, dtype=np.float64) 
 
	threads_per_block = TPB
	blocks_per_grid = (len(x) + threads_per_block - 1) // threads_per_block # always have enough blocks to cover all threads
	
	# initialize events for time recording
	eval_duration_start = cuda.event()
	eval_duration_stop = cuda.event()
 
 
	eval_duration_start.record()
	mono_kernel[blocks_per_grid, threads_per_block](d_out, d_x)	
	cuda.synchronize() 
	eval_duration_stop.record()
	eval_duration = cuda.event_elapsed_time(eval_duration_start, eval_duration_stop)
	
	transfer_duration_start = time()	
	y = d_out.copy_to_host()
	transfer_duration_stop = time()
 
	transfer_duration = (transfer_duration_stop - transfer_duration_start)*1000
 
	return(eval_duration, transfer_duration, y)

@cuda.jit('void(f8[:], f8[:])')
def gridstride_kernel(d_out, d_x):
	'''
	Monolithic kernel to compute arrays of s(x)
	Args:
		d_out: float device array to store output
		d_x: float device input array
	Returns: None
	'''
	# get grid starting position for this thread
	start_idx = cuda.grid(1)

	# get the total number of threads in the entire grid (the stride)
	stride = cuda.gridsize(1)

	# get the total size of the input/output array
	array_size = d_x.shape[0]

	# Use a grid-stride loop
	current_idx = start_idx # Start at the thread's initial position
	while current_idx < array_size:
		# do computation for this element
		d_out[current_idx] = s(d_x[current_idx]) # Using current_idx

		# move to the next element this thread is responsible for
		current_idx += stride # Use current_idx and add stride

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
	
	d_x = cuda.to_device(x) # copy input array to gpu memory
	
	# initialize output array for kernel function
	d_out = cuda.device_array(x.shape, dtype=np.float64)

	threads_per_block = TPB
	
	# Calculate blocks_per_grid based on SM count
	try:
		device = cuda.get_current_device()
		sm_count = getattr(device, 'MULTIPROCESSOR_COUNT', 1) # Get SM count, default to 1 if unavailable
		blocks_per_grid = sm_count * BLOCKSPERSM 
		print(f"Detected {sm_count} SMs, launching {blocks_per_grid} blocks.")
		print('\n') # separate the above print from the p1() outputs
	except cuda.cudadrv.error.CudaSupportError:
		print("CUDA device not found or properties unavailable. Using default block count.")
		# Fallback if SM count cannot be determined (e.g., no GPU)
		# Use normal calculation to get usable blocks_per_grid
		blocks_per_grid = (len(x) + threads_per_block - 1) // threads_per_block


	# initialize events for time recording
	eval_duration_start = cuda.event()
	eval_duration_stop = cuda.event()
	transfer_duration_start = cuda.event() # Use CUDA event for transfer timing
	transfer_duration_stop = cuda.event()  # Use CUDA event for transfer timing


	eval_duration_start.record()
	gridstride_kernel[blocks_per_grid, threads_per_block](d_out, d_x)
	eval_duration_stop.record()
	eval_duration_stop.synchronize() # Wait for kernel to finish before starting copy/timing

	transfer_duration_start.record() # Time the copy back
	y = d_out.copy_to_host() # Corrected copy
	transfer_duration_stop.record()
	transfer_duration_stop.synchronize() # Wait for copy to finish

	eval_duration = cuda.event_elapsed_time(eval_duration_start, eval_duration_stop)
	transfer_duration = cuda.event_elapsed_time(transfer_duration_start, transfer_duration_stop) # Calculate transfer time using events


	return(eval_duration, transfer_duration, y)

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

@cuda.jit('void(f8[:], f8[:], f8[:], f8)') # Added h_squared to signature
def ode_kernel(d_out, d_s, d_stencil, h_squared): # Added h_squared parameter
	'''
	Kernel to compute arrays of d2s/dx2 + s using global memory
	Args:
		d_out: float device array to store output
		d_s: float device input array
		d_stencil: float device array of stencil coefficients
		h_squared: float sample spacing squared
	Returns: None
	'''
	
	idx = cuda.grid(1)
	n = d_s.shape[0] # number of elements
	pi_squared = PI ** 2
	denom = 180.0 * h_squared

	# Check if the index is too close to the boundaries
	if idx < 3 or idx >= n - 3:
		d_out[idx] = 0.
	else:
		# Calculate the stencil sum
		stencil_sum = (d_stencil[0] * d_s[idx-3] +
					   d_stencil[1] * d_s[idx-2] +
					   d_stencil[2] * d_s[idx-1] +
					   d_stencil[3] * d_s[idx]   +
					   d_stencil[4] * d_s[idx+1] +
					   d_stencil[5] * d_s[idx+2] +
					   d_stencil[6] * d_s[idx+3])

		# Calculate d2s/dx2 approximation
		d2s_dx2 = stencil_sum / denom

		# Calculate the final value for the ODE check
		d_out[idx] = d2s_dx2 + pi_squared * d_s[idx]


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
	# Define the stencil coefficients
	stencil = np.array([2., -27., 270., -490., 270., -27., 2.], dtype=np.float64)
	h_squared = h**2

	# Copy input array and stencil to GPU memory
	d_s = cuda.to_device(s)
	d_stencil = cuda.to_device(stencil)
	# Initialize output array for kernel function
	d_out = cuda.device_array(s.shape, dtype=np.float64)

	threads_per_block = TPB
	blocks_per_grid = (len(s) + threads_per_block - 1) // threads_per_block

	# Initialize events for time recording
	eval_duration_start = cuda.event()
	eval_duration_stop = cuda.event()
	transfer_duration_start = cuda.event() # Use CUDA event for transfer timing
	transfer_duration_stop = cuda.event()  # Use CUDA event for transfer timing
	
	eval_duration_start.record()
	# Launch the kernel, passing h_squared as the new last argument
	ode_kernel[blocks_per_grid, threads_per_block](d_out, d_s, d_stencil, h_squared)
	eval_duration_stop.record()
	cuda.synchronize() # Wait for kernel to finish
 
	eval_duration = cuda.event_elapsed_time(eval_duration_start, eval_duration_stop)

	# Copy result back to host
	# Note: Instructions for 2a only ask for eval time and result array, not copy time.
	
 
	transfer_duration_start.record() # Time the copy back
	g_mem = d_out.copy_to_host()
	transfer_duration_stop.record()
	transfer_duration_stop.synchronize() # Wait for copy to finish
	transfer_duration = cuda.event_elapsed_time(transfer_duration_start, transfer_duration_stop) # Calculate transfer time using events
 
	# Return evaluation time and the result array
	return(eval_duration, transfer_duration, g_mem)

################# shared memory versions ##########

@cuda.jit('void(f8[:], f8[:], f8[:], f8)') # Added h_squared to signature
def sh_ode_kernel(d_out, d_s, d_stencil, h_squared): # Added h_squared parameter
	'''
	Kernel to compute arrays of d2s/dx2 + s using shared memory
	Args:
		d_out: float device array to store output
		d_s: float device input array
		d_stencil: float device array of stencil coefficients
		h_squared: float sample spacing squared
	Returns: None
	'''
	# Allocate shared memory array for the block
	# Size is Threads Per Block + 2 * Stencil Radius
	sh_s = cuda.shared.array(shape=(NSHARED,), dtype=float64)
	# Calculate indices
	idx = cuda.grid(1)       # Global thread index
	tx = cuda.threadIdx.x    # Thread index within the block
	n = d_s.shape[0]         # Total size of the input array

	# Load data into shared memory with boundary checks
	if idx < n:
		sh_s[tx + RAD] = d_s[idx]
	else:
		sh_s[tx + RAD] = 0.0  # Pad if outside bounds

	# Load halo elements
	if tx < RAD:
		# Load left halo
		left_idx = idx - RAD
		# Load right halo
		right_idx = idx + TPB
		sh_s[tx] = d_s[left_idx] if left_idx >= 0 else 0.0 # Load left halo element for this thread's corresponding position
		sh_s[tx + TPB + RAD] = d_s[right_idx] if right_idx < n else 0.0 # Load right halo element

	# Synchronize threads to ensure all shared memory is loaded
	cuda.syncthreads()

	# Perform stencil computation using shared memory within bounds
	if idx >= RAD and idx < n - RAD:
		# Calculate constants needed for the ODE check
		pi_squared = PI ** 2
		denom = 180.0 * h_squared

		# perform stencil on shared memory with radius offests. sh_s[tx + RAD] corresponds to d_s[idx]
		stencil_sum = (d_stencil[0] * sh_s[tx + RAD - 3] +
					   d_stencil[1] * sh_s[tx + RAD - 2] +
					   d_stencil[2] * sh_s[tx + RAD - 1] +
					   d_stencil[3] * sh_s[tx + RAD]     +
					   d_stencil[4] * sh_s[tx + RAD + 1] +
					   d_stencil[5] * sh_s[tx + RAD + 2] +
					   d_stencil[6] * sh_s[tx + RAD + 3])

		# Calculate d2s/dx2 approximation
		d2s_dx2 = stencil_sum / denom

		# Calculate the final value using the value from shared memory
		d_out[idx] = d2s_dx2 + pi_squared * sh_s[tx + RAD]

	# Handle the other boundary elements
	elif idx < n:
		d_out[idx] = 0.

	cuda.syncthreads()

def sh_ode_check(s, h):
	'''
	Compute array of values for d2s/dx2 + s using shared memory kernel
	Args:
		s: numpy float array of input values
		h: float sample spacing
	Returns:
		float evaluation time in ms
		float data transfer time in ms
		float array of function values
	'''
	# Define the stencil coefficients
	stencil = np.array([2., -27., 270., -490., 270., -27., 2.], dtype=np.float64)
	h_squared = h**2

	# Copy input array and stencil to GPU memory
	d_s = cuda.to_device(s)
	d_stencil = cuda.to_device(stencil)
	# Initialize output array for kernel function
	d_out = cuda.device_array(s.shape, dtype=np.float64)

	threads_per_block = TPB
	# Use monolithic launch strategy (like ode_check)
	blocks_per_grid = (len(s) + threads_per_block - 1) // threads_per_block

	# Initialize events for time recording
	eval_duration_start = cuda.event()
	eval_duration_stop = cuda.event()
	transfer_duration_start = cuda.event()
	transfer_duration_stop = cuda.event()

	eval_duration_start.record()
	# Launch the shared memory kernel, passing h_squared
	sh_ode_kernel[blocks_per_grid, threads_per_block](d_out, d_s, d_stencil, h_squared)
	eval_duration_stop.record()
	cuda.synchronize()

	eval_duration = cuda.event_elapsed_time(eval_duration_start, eval_duration_stop)

	# Time the copy back to host
	transfer_duration_start.record()
	sh_mem = d_out.copy_to_host()
	transfer_duration_stop.record()
	transfer_duration_stop.synchronize() # Wait for copy to finish
	transfer_duration = cuda.event_elapsed_time(transfer_duration_start, transfer_duration_stop)

	# Return evaluation time, transfer time, and the result array
	return(eval_duration, transfer_duration, sh_mem)

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

