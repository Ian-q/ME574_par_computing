import numpy as np
import matplotlib.pyplot as plt

def f(x:float, a:float) -> float:
    '''
    Logistic map
    Args:
        a: float parameter value
        x: float input value
    Returns:
        float value of logistic map function
    '''
        
    return a * x * (1 - x) # Logistic map function
    

def trajectory(f, x0:float, a:float, n:int) -> np.ndarray: 
    '''
    Compute an array of successive map iterates.
    Args:
        f: name of mapping function
        x0: float initial value
        a: float parameter value
        n: int number of iterations
    Returns:
        float numpy array of successive iterate values
    '''
    arr = np.zeros(n)  # Initialize an array to store the iterates
    arr[0] = x0  # Set the first entry to the initial value x0

    # Compute the each iteration using the mapping function f()
    for i in range(1, n):
        arr[i] = f(arr[i - 1], a)

    return arr

def spiderize(v:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute array of points for spiderweb plot
    Args:
        v: float numpy array of successive iterates
    Returns:
        x_spider: float numpy array of x coordinates
        y_spider: float numpy array of y coordinates
    '''
    
    # number of points
    n = len(v)

    # Each iterate (except the last) requires 2 points
    x_spider = np.zeros(2*n - 1)
    y_spider = np.zeros(2*n - 1)
    
    # initial point on the diagonal y=x
    x_spider[0] = v[0]
    y_spider[0] = v[0]
    
    # Build the zigzag pattern for the spiderweb
    for i in range(n - 1):
        # Move vertically from (x_i, x_i) to (x_i, x_{i+1})
        x_spider[2 * i + 1] = v[i]      # x-coordinate stays the same
        y_spider[2 * i + 1] = v[i + 1]  # y-coordinate moves to the next iterate

        # Move horizontally from (x_i, x_{i+1}) to (x_{i+1}, x_{i+1})
        x_spider[2 * i + 2] = v[i + 1]  # x-coordinate moves to the next iterate
        y_spider[2 * i + 2] = v[i + 1]  # y-coordinate stays the same
    
    return (x_spider, y_spider)

def iterate(f, x:float, a:float, k:int) -> np.ndarray:
    '''
    Compute an array of values for the k^th iteration of a mapping function
    Args:
        f: name of mapping function
        x: numpy array of float input values
        a: float parameter value
        k: int iteration number
    Returns:
        numpy array of k^th iterate values
    '''
    y = x.copy()
    
    # loop over all iterations
    for _ in range(k):
        new_y = np.zeros_like(y)
        
        # in each iteration, go over each index in y, and apply f()
        for i, val in enumerate(y):
            new_y[i] = f(val, a)
            
        y = new_y  # Update y with the new values
    return y
        

    
        
def p1():
    '''
    Test code for problem 1
    '''
    print("Start p1().")
    param = np.array([0.9, 1.9, 2.9, 3.1, 3.5, 3.57]) #example parameter values
    m = 21 # number of sample points
    x = np.linspace(0,1,m) #array of equally space points on [0,1]

    for j in range(2): # for the first 2 parameter values
        a = param[j]
        y = np.zeros(m) #array for storing computed values of the map
        for i in range(m): #for each entry in the array x
            y[i] =f(x[i],a) #store map value in corresponding entry in array y
        #visualize the map
        plt.plot(x,x, label='y=x') #45-degree line
        plt.plot(x,y, label='y=f(x)') #map data
        #label your axes (so your plot is not "information free")
        plt.xlabel('X')
        plt.ylabel('F')
        plt.legend()
        plt.title(f'f(x) for a = {a:.2f}')
        plt.show() #show the plot

    n = 1<<6 #trajectory length
    for j in range(param.shape[0]):
        a = param[j]
        x0 = 0.2
        iter_history = trajectory(f, x0, a, n)
        plt.plot(iter_history)
        plt.ylim([0,1])
        #label your axes (so your plot is not "information free")
        plt.xlabel('Iterations')
        plt.ylabel('$X_k$')
        plt.title(f'Iterate trajectory for a = {a:.2f}')
        plt.show() #show the plot

    print("How did the plot change?") 
    print('How will that affect "steady-state" behavior?')
    print('Inspect each of the following plots and notice how steady-state behavior changes.')


def p2():
    print("Start p2().")
    m = 21 # number of sample points
    x = np.linspace(0,1,m) #array of equally space points on [0,1]
    param = np.array([0.9, 1.9, 2.9, 3.1, 3.5, 3.57]) #example parameter values

    n = 1<<6 #trajectory length
    for j in range(param.shape[0]):
        a = param[j]
        x0 = 0.2
        iter_history = trajectory(f, x0, a, n)
        spider_x, spider_y = spiderize(iter_history)
        plt.plot(x,x) #45-degree line
        plt.plot(spider_x[6:-1], spider_y[6:-1]) #plot behavior after transient
        plt.ylim([0,1])
        #label your axes (so your plot is not "information free")
        plt.xlabel('X')
        plt.ylabel('$F(X)$')
        plt.title(f'Spiderweb: a = {a:.2f}')
        plt.show() #show the plot

def p3():
    print("Start p3().")
    k = 16 #sample iteration number
    m = 1024 # number of sample points
    x = np.linspace(0,1,m) #array of equally space points on [0,1]
    param = np.array([0.9, 1.9, 2.9, 3.1, 3.5, 3.57]) #example parameter values

    for j in range(param.shape[0]):
        a = param[j]
        kth_iter = np.zeros_like(x)
        kth_iter = iterate(f, x, a, k )
        plt.plot(x,x) #45-degree line
        plt.plot(x, kth_iter) #plot behavior after transient
        plt.ylim([0,1])
        #label your axes (so your plot is not "information free")
        plt.xlabel('X')
        plt.ylabel('$f^k(X)$')
        plt.title(f'Iterate {k:2d} for a = {a:.2f}')
        plt.show() #show the plot


if __name__ == '__main__':
    p1()
    p2()
    p3()