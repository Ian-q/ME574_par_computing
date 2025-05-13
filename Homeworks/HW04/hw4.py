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
	# INSERT CODE HERE
	pass

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
	# INSERT CODE HERE
	pass

@cuda.jit
def dist_kernel(d_out, d_x, d_y, eps, steps, h):
	# INSERT CODE HERE
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
def sum_reduce(a,b):
	# INSERT CODE HERE
	pass

@cuda.jit
def simpson_kernel(d_contribs, d_v):
	# INSERT CODE HERE
	pass

def par_simpson(v, h):
	'''
	Compute composite Simpson's quadrature estimate from uniform function sampling
	Args:
		v: 1D float numpy array of sampled function values
		h: float sample spacing
	Return:
		Float quadrature estimate
	'''
	# INSERT CODE HERE
	pass


def p2():
	print("\n\nProblem 2")
	n = 100000
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

# INSERT KERNEL CODE HERE

def jacobi_red_black(u, iter_count):
	# INSERT CODE HERE (INCLUDING KERNEL CALL)
	pass


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
	