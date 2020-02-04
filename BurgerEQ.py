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
    wx = params[:30]
    wt = params[30:60]
    b0 = params[60:90]
    w1 = params[90:120]
    b1 = params[120]
    h = sigmoid(x*wx + t*wt + b0)
    o = np.sum(h*w1) + b1
    return o

@jit
def loss(params, xai, tau, Ax, Bx, Bt):
    eq = dfdt_vect(params, xai, tau)/Bt + f_vect(params, xai, tau)*dfdx_vect(params, xai, tau)/Bx - 0.01*d2fdx2_vect(params, xai, tau)/(Bx*Bx)
    bc1 = f_vect(params, xai-xai - 1, tau)
    bc2 = f_vect(params, xai-xai + 1, tau)
    bc3 = f_vect(params, xai, tau-tau-1) - (xai*Bx + Ax)*(1-(xai*Bx + Ax))
    return (np.mean(eq**2) + np.mean(bc1**2) + np.mean(bc2**2) + np.mean(bc3**2))

def nest_momentum(xai, tau, momentum, lr, Ax, Bx, Bt):
    global params
    global velocity
    gradient = grad_loss(params + momentum*velocity, xai, tau, Ax, Bx, Bt)
    velocity = momentum*velocity - lr*gradient
    params += velocity

##########
## Main ##
##########

key = random.PRNGKey(0)
params = random.normal(key, shape=(121,)) * 0.5

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
t_values = np.linspace(0, 10, num=40)

x = []
t = []

for i in range(len(x_values)):
    for j in range(len(t_values)):
        x.append(x_values[i])
        t.append(t_values[j])

x = np.asarray(x)
t = np.asarray(t)

# Normalization
Ax = (max(x) + min(x)) / 2
Bx = (max(x) - min(x)) / 2
xai = (x - Ax)/Bx

At = (max(t) + min(t)) / 2
Bt = (max(t) - min(t)) / 2
tau = (t - At)/Bt

#-- Training parameters --#

#SGD momentum
epochs = 100000
learning_rate = 0.001
momentum = 0.99
velocity = 0.

#-- Training phase --#

for epoch in range(epochs):
    if epoch % 100  == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, xai, tau, Ax, Bx, Bt)))
    nest_momentum(xai, tau, momentum, learning_rate, Ax, Bx, Bt)

#-- Plotting and comparison with analytical solution --#

x_values = np.linspace(0, 1, num=100)
t_values = np.linspace(0, 10, num=100)

x = []
t = []

for i in range(len(x_values)):
    for j in range(len(t_values)):
        x.append(x_values[i])
        t.append(t_values[j])

x = np.asarray(x)
t = np.asarray(t)

Ax = (max(x) + min(x)) / 2
Bx = (max(x) - min(x)) / 2
xai = (x - Ax)/Bx

At = (max(t) + min(t)) / 2
Bt = (max(t) - min(t)) / 2
tau = (t - At)/Bt

fig = plt.figure()
ax = plt.axes(projection='3d')
plot = ax.scatter3D(x, t, f_vect(params, xai, tau), c = f_vect(params, xai, tau))
fig.colorbar(plot)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f(x,t)')
plt.show()
