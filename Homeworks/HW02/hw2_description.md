# ME 574 Spring 2025 Homework 2

1. Consider the problem of evaluating a function at an array of input points.

    a) In the file `hw2.py`, fill in the code for `sample_f21(xmin, xmax, n)` that computes an array of sampled values of the function `f(x)` at `n` input values equally spaced across the interval $[x_{min}, x_{max}]$.

    b) Fill in code for `sample_f21_parallel(xmin, xmax, n)` that computes the sample values in parallel using a grid instead of an iterative loop; i.e. by launching a kernel function. Your submitted code for this problem should include:
    - the completed code for `sample_f21_parallel`.
    - the kernel function that is launched by `sample_f21_parallel`.
    - the device function to be called from the kernel.
    </br></br>

    c)  When your code is running, execute the function `p1()` that tests your function definitions on the function 
    $$f(x) = 3 \pi^4 x^2 +ln(x-\pi)^2$$
     discussed in Schorghofer Problem 2.1. Note that finding the interesting feature of the function with a uniform sampling on $[0,4]$ requires a LOT of sample points and a significant amount of time for serial evaluation. (You may find the magnifying glass tool in the plot window to be helpful to see what is going on near $x=\pi$.) This very simple example already provides a motivation for parallelism.

2. Implement `time_f21` and `time_f21_parallel` that return the time to compute the array of values for your serial implementation of `sample_f21` (using `time.time()`) and `sample_f21_parallel` (using cuda event timing). The timing values will provide an estimate of the "acceleration" you achieved by parallelizing.

3. In Homework 1, you should have seen that, after repeated applications, even very simple maps can produce complicated plots with numerous extrema. In such cases, a large number of sample points are needed to capture the salient features of the resulting function. 

    The underlying computation involves the equivalent of a doubly-nested loop: 
    - one loop that iterates over the entries in the array of initial input values.
    - one loop that iterates to repeatedly apply the mapping function to the result of the previous iteration.

    In one of these loops, the computation for each index value depends on the result of other iterations; such a loop is not immediately parallelizable.

    In the other loop, the computation for each index value is independent of the result of other index values. This is the loop that is best-suited for parallelization.

    The file `hw2.py` contains a version of the code for the function `iterate` that computes sample values for the $k^{th}$ iterate of the logistic map `f` using a doubly-nested loop.

    Your task for this problem is to fill in code for the parallel version `iterate_parallel`. Subtasks will include:
- Identifying the loop that is appropriate for parallelization.
- Replacing that loop with a call to launch a parallel kernel function.
- Definition of the kernel function.

    When you are done writing the code, execute the file including the function `p3()` that calls both `iterate` and `iterate_parallel` and plots the results for comparison. Inspect the output to verify that your code works properly.


