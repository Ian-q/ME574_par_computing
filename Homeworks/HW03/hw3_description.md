# ME 574 Spring 2025: Homework 3 description

These instructions describe the code that you should enter in your version of `hw3.py` (which is your deliverable for this assignment).

1. This problem involves comparing performance when implementing a problem fitting the `map` pattern in 3 ways: serial, parallel (with monolithic kernel launch), parallel with grid-stride loop.

   a) (2 point) Fill in code to implement `timed_serial(x)` that computes values of $s(x) = \sin(\pi x)$ for an array of input values evenly spaced on $[0,1]$. Your implementation should return a 2-tuple including the python evaluation timing (in  $ms$ - remember to multiply by 1000 since `time()` uses units of seconds) and the array of computed values.

   b) (3 points) Fill in code to implement `timed_parallel_(x)` that computes values of $s(x)$ for an array of input values in parallel by calling `mono_kernel()` (which you should also implement) that uses a monolithic kernel launch (with each thread computing the function value for a single entry in the input array). Your implementation should return a 3-tuple including (i) the CUDA event timing for the array evaluation, (ii) the CUDA event timing for transferring the computed array to the host, and (iii) the array of computed values.

   c) (5 points) Fill in code to implement `timed_gridstride(x)` that computes values of $s(x)$ for an array of input values in parallel by calling `gridstride_kernel()` (which you should also implement) that uses a kernel with a grid stride loop with 4 blocks per SM. Your implementation should return a 3-tuple including (i) the CUDA event timing for the array evaluation, (ii) the CUDA event timing for transferring the computed array to the host, and (iii) the array of computed values.

> When you get your code working and run `p1()`, do take a minute to look at the information that is plotted and/or printed to the terminal. First, look at the plots to make sure that the results from using the different (monolithic and gridstride) kernels actually agree. Secondly, look at the "acceleration" estimates. How much did basic parallelization speed up your computations? How much additional speedup was achieved by using a gridstride loop?


2. In this problem, you implement a stencil computation to estimate the second derivative of the function $s(x)$ and to check that $s(x)$ provides a reasonable solution of the ordinary differential equation $~~\frac{d^2s}{dx^2} + \pi^2 s = 0$.

	a) (3 points) Fill in code to implement `ode_check(x)` that computes an array of values of $\frac{d^2s}{dx^2} + \pi^2 s$ where the $2^{nd}$ derivative is estimated using a central difference approximation with a stencil of radius 3. The corresponding stencil to estimate $\frac{d^2 f}{dx^2}$ is:
	$$\{2, -27, 270, -490, 270, -27, 2\}/(180*1.0*h**2)$$ 
	`ode_check()` should execute in parallel calling `ode_kernel()` that launches a monolithic kernel (where each thread computes 1 entry in the output). __To avoid complications associated with boundaries, just return the value zero in the last 3 entries on either end of the array__ (where the stencil would extend beyond the array bounds). `ode_check(x)` returns a 2-tuple including (i) CUDA event timing for the array evaluation and (ii) the array of computed values.

	b) (7 points) Repeat 3a, but with `sh_ode_check(x)` that has the same inputs and outputs as `ode_check(x)` but calls `sh_ode_kernel()` (which you should also implement) that uses a shared memory array.

> Again, after you get your code working and execute `p2()`, take a minute to look at the details of the output. Verify that the results from using the kernel with global memory acetually agree with the results of the kernel using shared memory, and not how the accelerations from basic parallelization compare with the additional acceleration gained by using a shared memory array.

