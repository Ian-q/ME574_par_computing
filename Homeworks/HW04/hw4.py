# ME 574 Homework 4 Solutions
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit
from math import sqrt, sin, sinh

EPS = 2.0
TX, TY = 16, 16
msPerSec = 1000
TPB = 32

@jit #@cuda.jit(device=True)
def f(x:float, y:float, eps:float) -> (float, float):
	'''
	Compute the rate vector for the vdP system
	Args:
		x,y: float phase-plane coords. (displacement and velocity)
		eps: float parameter value
	Returns: 
		rhs: float 2-tuple of rates of change for x and y
	'''
	return y, -x + eps*(1-x*x)*y

@jit #@cuda.jit(device=True)
def vdp_rk2_step(x0:float, y0:float, eps:float, h:float):
	'''
	Compute updated state of the van der Pol system after 1 time step
	Args:
		x0,y0: float phase-plane coords. (displacement and velocity)
		eps: float parameter value
		h: float timestep
	Returns: 
		x, y: float updated coordinates
	'''
	# slope now
	k1x, k1y = f(x0, y0, eps)
	
	# estimated state 1/2 timestep forward
	x_mid = x0 + (h/2)*k1x
	y_mid = y0 + (h/2)*k1y
 
	# slope at estimated 1/2 timestep forward
	k2x, k2y = f(x_mid, y_mid, eps)
	
	# extrapolate state 1 timestep forward using 1/2 timestep	
	x = x0 + h*k2x
	y = y0 + h*k2y
 
	return (x, y)

@cuda.jit(device=True)
def rk2_dist(x0:float, y0:float, eps:float, steps:int, h:float):
	'''
	Use RK2 method to compute the solution of the van der Pol equation
	Args:
		x0, y0: float initial displacement and velocity
		eps: float parameter value
		steps: int step count
		h: float time step
	Returns:
		Float final distance from origin of phase plane: sqrt(x*x+y*y)
	'''
	# remap vars
	x = x0
	y = y0
	
	# calcualte d and v for all steps
	for _ in range(steps):
		x, y = vdp_rk2_step(x, y, eps, h)
 
	return sqrt(x*x+y*y)

@cuda.jit
def dist_kernel(d_out, d_x, d_y, eps, steps, h):
	# get thread index
	ix, iy = cuda.grid(2)
 
	# check that thread is within bounds
	nx = d_out.shape[0]
	ny = d_out.shape[1]
	if (ix > nx) or (iy > ny):
		return

	# from the thread indeces, get initial x and y values
	x_initial = d_x[ix]
	y_initial = d_x[iy]
 
	# get distance
	distance = rk2_dist(x_initial, y_initial, eps, steps, h)
	
	# store output
	d_out[ix, iy] = distance
	pass

def dist(x , y, eps, steps, h):
	'''
	Compute solutions of vdP equation for a grid of initial conditions
	Args:
		x,y: 1D arrays of initial displacement and velocity
		eps: float parameter value
		steps: int number of time steps in the simulation
		h: float time step
	Returns:
		2D numpy float array of final distance from origin of phase plane
	'''
	d_x = cuda.to_device(x)
	d_y = cuda.to_device(y)
	nx = x.size
	ny = y.size
	d_out = cuda.device_array([nx,ny])
	threads = TX,TY
	blocks = (nx+TX-1)//TX, (ny+TY-1)//TY
	dist_kernel[blocks, threads](d_out, d_x, d_y, eps, steps, h)
	return d_out.copy_to_host()

def p1():
	print("\nProblem 1")
	RX,RY = 6., 6.
	NX,NY = 1024, 1024
	steps = 5000
	h = 0.01
	# Run a single simulation to check that RK2 code is working properly
	# x = 2.1
	# y = 0.2
	# vals = np.zeros(steps)
	# for i in range(steps):
	# 	vals[i] = x
	# 	x, y = vdp_rk2_step(x, y, EPS, h)
	# plt.plot(vals)
	# plt.show()
	x = np.linspace(-RX,RX,NX)
	y = np.linspace(-RY,RY,NY)
	d0 = dist(x,y,EPS,steps,h)
	plt.imshow(np.fliplr(d0.T), extent=[-RX,RX,-RY,RY], vmin=0., vmax = 2.5)
	cb = plt.colorbar()
	plt.title('Distance from Equilibrium')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.show()
	d1 = dist(x,y,EPS,steps,-h)
	plt.imshow(np.fliplr(d1.T), extent=[-RX,RX,-RY,RY], vmin=0., vmax = 2.5)
	cb = plt.colorbar()
	plt.title('Distance from Equilibrium')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.show()


#================= START PROBLEM 2 =========================

@cuda.reduce
def sum_reduce(a, b):
	# Device reduction for summing elements,
	# note: make sure to use gpu arrays because gpu
	return a + b

@cuda.jit
def simpson_kernel(d_contribs, d_v, h):
	# Each thread computes one Simpson's rule panel contribution.
	i = cuda.grid(1)
	n = d_v.size

	# There are (n-1)//2 panels for n points (n odd)
	if i < (n - 1) // 2:
		vi = 2 * i + 1  # odd index, panel center
		contrib = (h / 3.0) * (d_v[vi - 1] + 4.0 * d_v[vi] + d_v[vi + 1])
		d_contribs[i] = contrib

def par_simpson(v, h):
	'''
	Compute composite Simpson's quadrature estimate from uniform function sampling
	Args:
		v: 1D float numpy array of sampled function values
		h: float sample spacing
	Return:
		Float quadrature estimate
	'''
	# Ensure the input size is odd for Simpson's rule
	n = v.size
	if n % 2 == 0:
		raise ValueError("Simpson's rule requires an odd number of points.")

	# Number of panels
	num_panels = (n - 1) // 2

	# Allocate device arrays
	d_v = cuda.to_device(v)
	d_contribs = cuda.device_array(num_panels, dtype=np.float64)

	# Launch kernel: one thread per panel
	threads_per_block = TPB
	blocks_per_grid = (num_panels + threads_per_block - 1) // threads_per_block
	simpson_kernel[blocks_per_grid, threads_per_block](d_contribs, d_v, h)

	# Use device reduction to sum the panel contributions
	integral = sum_reduce(d_contribs)

	return float(integral)


def p2():
	print("\n\nProblem 2")
	n = 100001 # NOTE: I CHANGED THIS SO THAT IT WOULD WORK BECAUSE SIMPSON'S RULE WANTS AN ODD NUMBERED SIZE
	xmin, xmax = 0, 2*np.pi
	h = (xmax-xmin)/(n-1)
	x = np.linspace(xmin, xmax, n)
	v = np.sin(100*x) * np.sin(100*x)
	result = par_simpson(v, h)
	print("Integral value: ", result)

#================ START PROBLEM 3 ============================
# A version of the jacobi_global code is included as a starting point
@cuda.jit("void(f8[:,:], f8[:,:])")
def updateKernel(d_v, d_u):
	i,j = cuda.grid(2)
	dims = d_u.shape
	if i >= dims[0] or j >= dims[1]:
		return

	if i>0 and j>0 and i<dims[0]-1 and j<dims[1]-1:
		d_v[i, j] = (d_u[i-1, j] + d_u[i+1,j] + d_u[i,j -1] + d_u[i, j+1])/4. 

def update(u, iter_count):
	d_u = cuda.to_device(u)
	d_v = cuda.to_device(u)
	dims = u.shape
	gridDims = ((dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB)
	blockDims = (TPB, TPB)

	for k in range(iter_count):
		updateKernel[gridDims, blockDims](d_v, d_u)
		updateKernel[gridDims, blockDims](d_u, d_v)
	return d_u.copy_to_host()

@cuda.jit("void(f8[:,:], int32)")
def red_black_update_kernel(data_array, color_to_update):
	"""
	CUDA kernel for red-black Jacobi update.
	Updates points of a specific color in-place.
	color_to_update: 0 for red points ( when (i+j)%2 == 0), 1 for black points (when (i+j)%2 == 1).
	"""
	i, j = cuda.grid(2)
	num_rows, num_cols = data_array.shape

	# Check if thread is within array bounds
	if (i > 0) and (i < num_rows - 1) and (j > 0 and j < num_cols - 1):
		# Determine the "color" of the current point (i, j)
		# (i + j) % 2 == 0 is "red" (matches color_to_update == 0)
		# (i + j) % 2 == 1 is "black" (matches color_to_update == 1)
		point_color = (i + j) % 2
		
		# If this point's color matches the color currently being updated:
		if point_color == color_to_update:
			# Perform Jacobi update for the selected color point using the average of its four neighbors
			data_array[i, j] = (data_array[i - 1, j] + data_array[i + 1, j] + data_array[i, j - 1] + data_array[i, j + 1]) / 4.0

def jacobi_red_black(u, iter_count):

	# allocate gpu memory
	d_u = cuda.to_device(u)
	rows, cols = u.shape
	
	# Configure kernel launch parameters (using the global TPB)
	block_dim = (TPB, TPB) 
	grid_dim = ((rows + TPB - 1) // TPB, (cols + TPB - 1) // TPB)

	# Alternate between red and black updates
	for _ in range(iter_count):
		# Update "red" points (color_to_update = 0)
		red_black_update_kernel[grid_dim, block_dim](d_u, 0)
		
		# Update "black" points (color_to_update = 1)
		red_black_update_kernel[grid_dim, block_dim](d_u, 1)
		
	u_updated = d_u.copy_to_host() # Copy the result back to the host
	return u_updated

def p3():
	NX, NY = 101, 101
	iters = NX*NX  #//2
	PI = np.pi
	TPB = 8
	#Compute exact solution
	exact = np.zeros(shape=[NX,NY])
	for i in range(NX):
		for j in range(NY):
			exact[i,j]= sin(i*PI/(NX-1)) * sinh(j*PI/(NY-1))/sinh(PI)

	#set initial guess satisfying BCs (with zero elsewhere)
	u = np.zeros(shape=[NX,NY])
	for i in range(NX):
		u[i,NX-1]= sin(i*PI/(NX-1))
		
	u = jacobi_red_black(u, iters)
	error = np.max(np.abs(u-exact))
	print("NX = %d, iters = %d => max error: %5.2e"  %(NX, iters, error))
	xvals = np.linspace(0., 1.0, NX)
	yvals = np.linspace(0., 1.0, NY)
	X,Y = np.meshgrid(xvals, yvals)
	levels = [0.025, 0.1, 0.25, 0.50, 0.75]
	plt.contourf(X,Y,exact.T, levels = levels)
	plt.contour(X,Y,u.T, levels = levels,
		colors = 'r', linewidths = 4)
	plt.axis([0,1,0,1])
	plt.show()

#================ main =======================================
def main():
	p1()
	p2()
	p3()

if __name__ == '__main__':
	main()
	