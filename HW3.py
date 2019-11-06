
# %% [markdown]
# # Homework 3

# %%
import pandas as pd
import numpy as np
import sympy as sym
from sympy import Symbol, cos, sin, lambdify
import matplotlib.pyplot as plt
import pdb
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


# %%
x1  = sym.Symbol('x1')
x2 = sym.Symbol('x2')
function = (5*x1-x2)**4+((x1-2)**2)+(x1-2*x2)+12
f = lambdify([[x1,x2]], function, 'numpy')
# plot the function
f_2 = lambdify([x1,x2], function, 'numpy')


# %%
x = np.arange(6.0,7.0,0.1)
y = np.arange(32.0,33.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = f_2(X,Y) # evaluation of the function on the grid

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# %% [markdown]
# ## Cyclic Coordinate Search

# %%
def CyclicCoordinateSearch(f, x0, epsilon):
    """
    Cyclic Coordinate Search method.
    Parameters
    ----------
    f : lambda expression The function to be evaluated
    x0 : numpy.array Starting coordinates
    epsilon : float Epsilon value determined to check optimality
    Returns
    ---------
    x1 : numpy.array The found x* vector
    k : integer Number of iterations
    """
    k = 0
    n = len(x0)
    while(True):
        y0 = x0
        for j in range(n):
            d = np.zeros(n)
            d[j] = 1
            # exact line search:
            # minimize y1 = f(y0+alpha*d)
            y0 = y1
        x1 = y0
        k += 1
        if(np.linalg.norm(x1-x0) < epsilon):
            return x1, k
        else:
            x0 = x1


# %%


# %% [markdown]
# ## Hook & Jeeves Method

# %%
# code goes to here

# %% [markdown]
# ## Simplex Search

# %%
def SimplexSearch():
    x=np.zeros(shape=(3,2))
    x_1=np.array([-2,15])#initial
    x_2=np.array([-8,10])#initial
    x_3=np.array([0,0])#initial
    x[0]=x_1
    x[1]=x_2
    x[2]=x_3 
    def compute_f_values(a):
        f_values=np.zeros(a.shape[0])    
        for i in range(a.shape[0]):
            f_values[i]=f(a[i])
        return f_values    
    epsilon=1
    alpha=1
    beta=0.5
    gamma=2
    
    while(True):    
        sum_value=0
        f_values=compute_f_values(x)#function values of x_matrix
        x_h=x[np.argmax(f_values)] #the worst point
        x_l=x[np.argmin(f_values)] #the best point
        index_to_go=np.argmax(f_values) #index of x_h
        mean_x=np.delete(x, index_to_go, 0)#x matrix rather than x_h
        x_mean=np.mean(mean_x,axis=0)#compute the mean of x matrix rather than x_h
        f_values_mean_x=compute_f_values(mean_x)#function values of x_matrix except x_h
        
        x_r=x_mean+alpha*(x_mean-x_h) #reflection
        
        if f(x_l)>f(x_r): #the reflected point x_r happens to be better than the current best  
            x_e=x_mean+gamma*(x_r-x_mean) #Expansion
            if f(x_r)>f(x_e): #the expanded point x_e happens to be better than the current best x_r
                x[np.argmax(f_values)]=x_e
            else:             #the expanded point is not better than x_r so we replace x_h with x_r
                x[np.argmax(f_values)]=x_r
        
        else:
            if (np.max(f_values_mean_x))>=f(x_r):
                x[np.argmax(f_values)]=x_r
            else:
                if f(x_h)>f(x_r):
                    x_h_prime=x_r
                else:
                    x_h_prime=x_h
                
                x_c=x_mean+beta*(x_h_prime-x_mean) #contraction
                if f(x_c) <= f(x_h):
                    x[np.argmax(f_values)]=x_c
                else:
                    for i in range(3):
                        x[i]=x[i]+0.5*(x_l-x[i]) #shrink operation
        
        for i in range(3):
            sum_value+=(f(x[i])-f(x_mean))**2
        print(sum_value)
        if np.sqrt(sum_value)<epsilon:
            break
                
    return x_mean

