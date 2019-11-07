
# %% [markdown]
# # Homework 3

# %%
import pandas as pd
import numpy as np
from sympy import Symbol, cos, sin, lambdify


# %%
x1  = Symbol('x1')
x2 = Symbol('x2')
function = (5*x1-x2)**4+((x1-2)**2)+(x1-2*x2)+12
f = lambdify([[x1,x2]], function, 'numpy')


# %%
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# plot the function
x = np.arange(0,15,0.5)
y = np.arange(10,50,0.5)
X,Y = meshgrid(x, y) # grid of point
Z = f([X,Y]) # evaluation of the function on the grid

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("graph.png")
plt.show()

# %% [markdown]
# ## Exact Line Search

# %%
def BisectionMethod(f, a=-100,b=100,epsilon=0.005) :
    iteration=0
    while (b - a) >= epsilon:
        x_1 = (a + b) / 2
        fx_1 = f(x_1)
        if f(x_1 + epsilon) <= fx_1:
            a = x_1
        else:
            b = x_1
        iteration+=1
    x_star = (a+b)/2
    return x_star


# %%
def ExactLineSearch(f, x0, d):
    alpha = Symbol('alpha')
    function_alpha = f(np.array(x0)+alpha*np.array(d))
    f_alp = lambdify(alpha, function_alpha, 'numpy')
    alp_star = BisectionMethod(f_alp)
    return alp_star

# %% [markdown]
# ## Cyclic Coordinate Search

# %%
def np_str(x_k):
    '''
    Used to convert numpy array to string with determined format
    '''
    return np.array2string(x_k, precision=2, separator=',')


# %%
def CyclicCoordinateSearch(f, x0, epsilon):
    x0 = np.array(x0)
    x_array = [x0]
    k = 0
    n = len(x0)
    res_array = []
    while(True):
        y0 = np.copy(x_array[k])
        for j in range(n):
            d = np.zeros(n)
            d[j] = 1
            alpha = ExactLineSearch(f, y0, d)
            y1 = y0 + alpha*d
            res_array.append([k, np_str(x_array[k]), f(x_array[k]),j, str(d),np_str(y0), alpha, np_str(y1)])
            y0 = y1
        x_array.append(y1)
        k += 1
        if(np.linalg.norm(x_array[k]-x_array[k-1]) < epsilon):
            res_array.append([k, np_str(x_array[k]), f(x_array[k])])
            result_table = pd.DataFrame(res_array, columns=['k' ,'x^k', 'fx^k', 'j','d^j','y^j','a^j', 'y^j+1'])
            return result_table

# %% [markdown]
# **Solution set 1:**
# *   x^0 = \[0, 0\]
# *   Epsilon = 0.01

# %%
output1 = CyclicCoordinateSearch(f,[0,0],0.01)
output1

# %% [markdown]
# **Solution set 2:**
# *   x^0 = \[10, 35\]
# *   Epsilon = 0.01

# %%
output2 = CyclicCoordinateSearch(f,[10,35],0.01)
output2


# %%
print(output2[-11:].to_latex(index=False,float_format='%.3f'))

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


# %%
SimplexSearch()

