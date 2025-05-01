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
    X = a * x * (1-x)
    return X

def trajectory(f, x0:float, a:float, n:int) -> np.ndarray: 
    '''
    Compute an array of successive map iterates
    Args:
        f: name of mapping function
        x0: float initial value
        a: float parameter value
        n: int number of iterations
    Returns:
        float numpy array of successive iterate values
    '''
    t = np.empty(n)
    t[0] = x0
    for i in range(1, n):
        t[i] = f(t[i - 1], a)
    return t
    

def spiderize(v:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute arrayof points for spiderweb plot
    Args:
        v: float numpy array of successive iterates
    Returns:
        x_spider: float numpy array of s coordinates
        y_spider: float numpy array of y coordinates
    '''
    x_spider = [v[0]]
    y_spider = [v[0]]
    for i in range(len(v) - 1):
        # Vertical segment: from (v[i], v[i]) to (v[i], v[i+1])
        x_spider.append(v[i])
        y_spider.append(v[i+1])
        # Horizontal segment: from (v[i], v[i+1]) to (v[i+1], v[i+1])
        x_spider.append(v[i+1])
        y_spider.append(v[i+1])
    return np.array(x_spider), np.array(y_spider)

def iterate(f, x: np.ndarray, a: float, k: int) -> np.ndarray:
    '''
    Compute an array of values for the k^th iteration of the mapping function.
    Args:
        f: mapping function
        x: numpy array of input values
        a: float parameter value
        k: int iteration number
    Returns:
        numpy array of the k^th iterate values
    '''
    y = x.copy()
    for _ in range(k):
        y = np.array([f(val, a) for val in y])
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
        plt.plot(x, x, label='y=x', color='r') #45-degree line
        plt.plot(x,y, label='y=f(x)', color='b') #map data
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
        plt.plot(iter_history, color='b')
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
        plt.plot(x,x, color='b') #45-degree line
        plt.plot(spider_x[6:-1], spider_y[6:-1], color='r') #plot behavior after transient
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
        plt.plot(x,x, color='b') #45-degree line
        plt.plot(x, kth_iter, color='r') #plot behavior after transient
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