import sys
import sys
import jax.numpy as np
from jax import random
from jax import grad
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

###############
## Functions ##
###############

def leakRelu(x, a):
    return np.where( x > 0, x, a*x)

def softplus(x):
    return np.log(1 + np.exp(x))

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def f(params, x, t):
    wx = params[:100]
    wt = params[100:200]
    b0 = params[200:300]
    w1 = params[300:400]
    b1 = params[400]
    h = softplus(x*wx + t*wt + b0)
    o = np.sum(h*w1) + b1
    return o

@jit
def loss(params, x, t):
    eq = dfdt_vect(params, x, t) + f_vect(params, x, t)*dfdx_vect(params, x, t) - 0.01*d2fdx2_vect(params, x, t)
    bc1 = f_vect(params, x-x, t)
    bc2 = f_vect(params, x-x + 1, t)
    bc3 = f_vect(params, x, t-t) - x*(1-x)
    return np.mean(eq**2) + np.mean(bc1**2) + np.mean(bc2**2) + np.mean(bc3**2)

def nest_momentum(x, t, momentum, lr):
    global params
    global velocity
    gradient = grad_loss(params + momentum*velocity, x, t)
    velocity = momentum*velocity - lr*gradient
    params += velocity


##########
## Main ##
##########

key = random.PRNGKey(0)
params = random.normal(key, shape=(401,)) * 0.07

#-- Setting up the functions and derivatives --#

dfdx = grad(f, 1)
dfdt = grad(f, 2)
d2fdx2 = grad(dfdx, 1)
f_vect = vmap(f, (None, 0, 0))
dfdx_vect = vmap(dfdx, (None, 0, 0))
dfdt_vect = vmap(dfdt, (None, 0, 0))
d2fdx2_vect = vmap(d2fdx2, (None, 0, 0))
grad_loss = jit(grad(loss, 0))

#-- Defining the domain of x, t, and bc1 --#

x_values = np.linspace(0, 1, num=40)
t_values = np.linspace(0, 5, num=40)

x = []
t = []

for i in range(len(x_values)):
    for j in range(len(t_values)):
        x.append(x_values[i])
        t.append(t_values[j])

x = np.asarray(x)
t = np.asarray(t)

#-- Training parameters --#

epochs = 500000
learning_rate = 0.005
momentum = 0.99
velocity = 0.

#-- Training phase --#

for epoch in range(epochs):
    if epoch % 100  == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, x, t)))
    nest_momentum(x, t, momentum, learning_rate)

#-- Plotting and comparison with analytical solution --#

x_values = np.linspace(0, 1, num=100)
t_values = np.linspace(0, 5, num=100)

x = []
t = []

for i in range(len(x_values)):
    for j in range(len(t_values)):
        x.append(x_values[i])
        t.append(t_values[j])

x = np.asarray(x)
t = np.asarray(t)

fig = plt.figure()
ax = plt.axes(projection='3d')
plot = ax.scatter3D(x, t, f_vect(params, x, t), c = f_vect(params, x, t))
fig.colorbar(plot)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f(x,t)')
plt.show()
