# ME 574 Spring 2025 Homework 4

1. The van der Pol equation:

$$x'' - \epsilon (1-x^2) x' + x = 0$$

is a classic example of an oscillator with a nonlinear damping term that can change sign so the "damper" can not only drain energy from the system (as expected from a linear damper) but can also pump energy into the system. Given the possibility of pumping energy into the system, it is interesting to consider the stability of the equilibrium solution $x(t) \equiv x'(t) \equiv 0$.

Converting the $2^{nd}$-order ODE to a first order system gives:

$$\begin{aligned}
x' &= y \\
y' &= -x + \epsilon (1-x^2) y
\end{aligned}$$

where $y$ is the velocity. 

We can now think of trajectories as residing in the 2D $xy-\negmedspace$ plane (a.k.a. the _phase plane_) which contains the equilibrium solution $x(t) \equiv y(t) \equiv 0$.

The most interesting stability questions involve:
- Local stability: 
"Do all initial conditions in the neighborhood of the equilibrium lead to solutions that stay near the equilibrium?"
- Global stability:  
"Do all initial conditions $\sout{\text{in the neighborhood of the equilibrium}}$ lead to solutions that $\sout{\text{stay}} \; end \; up$ near the equilibrium?"

It turns out that , for $\epsilon > 0$, the equilibrium solution at the origin is locally unstable (which you can show by analyzing the system linearized near the origin). That brings us to the question: "What is the long-term behavior? Do things just blow up toward infinity or does something else happen?"

The goal of this problem is to compute approximate solutions of the van der Pol equation to determine the long-term (steady-state) behavior. 

Your mission is to write code for the following functions:
- `vdp_rk2_step(x0, y0, eps, h)` that implements $2^{nd}$-order Runge-Kutta to compute the state of the van der Pol system after a single time step.
- `rk2_dist(x0, y0, eps, steps, h)` that computes the distance the final state lies from the equilibrium at the origin of the phase plane for a grid of starting conditions with coordinates `x0, y0`.
- `dist_kernel(d_out, d_x, d_y, eps, steps, h)` that gets called by the wrapper function `dist(x , y, eps, steps, h)` to compute the array of final distances arising from a grid of initial states `d_x[i], d_y[j]`.

The function `p1()` calls `dist()` and plots the grid of distance values (using `imshow()`). If your implementation is working correctly, you should see that this first plot indicates that all initial conditions lead to final distance values in the approximate interval $[1.25,2.5]$. Since this interval does not include $0$, the equilibrium at the origin must be unstable. Trajectories leave the vicinity of the origin, but where do they end up? To address that question, `p1()` creates a second plot based on array of simulations where the timestep is negative. This second plot should show a dark central region.

Rhetorical questions to consider:
- What can you draw from the second plot? 
- How do you interpret what is happening with points that lie outside the dark region? 

2. This problem combines stencils with reduction in the context of numerical integration (a.k.a. quadrature). Consider the problem of estimating the value of an integral based on a uniform sampling of function values using Composite Simpson's rule. Given an array `v` containing `2*n` sampled function values, the integral is estimated as the sum of panel contributions. 
The contribution from the panel spanning index values  `j-1,j,j+1` is given by the following formula:
 `(h/3)*(v[i-i] + 4*v[i] + v[i+1])`.

	Your task is to implement `par_simpson(v, h)` that computes the panel contributions in parallel using a global array and then to perform a reduction to sum the panel contributions to produce the integral estimate.

	Your code should include:
- Implementation of `par_simpson(v, h)`
- A kernel for parallel evaluation of panel contributions
- A reduction function to sum the panel contributions

The function `p2()` will call your implementation with test data sampled from a function with a known integral value.

3. Write code for `jacobi_red_black()` that implements the Jacobi iteration discussed in class. Your code should solve the Laplace equation using a single global memory device array and compute repeatable updates. (If you run your code multiple times, the sequence of updated matrices should not change.) Your complete code should include:
- the function `jacobi_red_black()`.
- a kernel function that it calls to perform the parallel computation.
- any device functions you want to call from your kernel.
