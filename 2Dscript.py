import sys
import jax.numpy as np
from jax import random
from jax import grad
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Functions

def softplus(x):
    return np.log(1 + np.exp(x))

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def f(params, x, t, bc1):
    wx = params[:80]
    wt = params[80:160]
    wbc = params[160:240]
    b0 = params[240:320]
    w1 = params[320:400]
    b1 = params[400]
    h = softplus(x*wx + t*wt + bc1*wbc + b0)
    o = np.sum(h*w1) + b1
    return o

@jit
def loss(params, x, t, bc1):
    eq = dfdx_vect(params, x, t, bc1) + dfdt_vect(params, x, t, bc1) - (3 * x) - t
    bc1_res = f_vect(params, x-x, t, bc1) - bc1
    return np.mean(eq**2) + np.mean(bc1_res**2)

##########
## Main ##
##########

key = random.PRNGKey(0)
params = random.normal(key, shape=(401,))

#-- Setting up the functions and derivatives --#

dfdx = grad(f, 1)
dfdt = grad(f, 2)
f_vect = vmap(f, (None, 0, 0, 0))
dfdx_vect = vmap(dfdx, (None, 0, 0, 0))
dfdt_vect = vmap(dfdt, (None, 0, 0, 0))
grad_loss = jit(grad(loss, 0))

#-- Defining the domain of x, t, and bc1 --#

x_values = np.linspace(-1, 1, num=40)
t_values = np.linspace(-1, 1, num=40)
bc1_values = np.linspace(1, 4, num=4)

x = []
t = []
bc1 = []

for i in range(len(x_values)):
    for j in range(len(t_values)):
        for k in range(len(bc1_values)):
            x.append(x_values[i])
            t.append(t_values[j])
            bc1.append(bc1_values[k])

x = np.asarray(x)
t = np.asarray(t)
bc1 = np.asarray(bc1)

#-- Training parameters --#

epochs = 1000
learning_rate = 0.0005
momentum = 0.99
velocity = 0.

#-- Training phase --#

for epoch in range(epochs):
    if epoch % 100  == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, x, t, bc1)))
    gradient = grad_loss(params + momentum*velocity, x, t, bc1)
    velocity = momentum*velocity - learning_rate*gradient
    params += velocity

#-- Plotting and comparison with analytical solution --#

x = []
t = []
bc1 = []
for i in range(len(x_values)):
    for j in range(len(t_values)):
        x.append(x_values[i])
        t.append(t_values[j])
        bc1.append(3)

x = np.asarray(x)
t = np.asarray(t)
bc1 = np.asarray(bc1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, t, f_vect(params, x, t, bc1))
#ax.scatter3D(x, t, f_vect(params, x, t, 3))
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f(x,t)')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, t,(x*x + t*x + 3))
#ax.scatter3D(x, t, f_vect(params, x, t, 3))
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f(x,t)')
plt.show()

#plt.plot(inputs, , label='exact')
#plt.plot(inputs, f_vect(params, inputs), label='approx')
#plt.legend()
#plt.show()
