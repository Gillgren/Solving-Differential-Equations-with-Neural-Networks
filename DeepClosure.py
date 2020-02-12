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

@jit
def w(w_params, v):
    w_w0 = w_params[:30]
    w_b0 = w_params[30:60]
    w_w1 = w_params[60:90]
    w_b1 = w_params[90]
    w_h = sigmoid(v*w_w0 + w_b0)
    w_o = np.sum(w_h*w_w1) + w_b1
    return w_o

@jit
def W_enc(f_emp, w, v_max, v_min):
    integrand = np.multiply(f_emp, w)
    mean_integrand = np.mean(integrand)
    return mean_integrand * (v_max-v_min)

@jit
def decoder(d_params, v, W_enc, U_1, U_2, U_3):
    d_v = d_params[:30]
    d_wenc = d_params[30:60]
    d_U1 = d_params[60:90]
    d_U2 = d_params[90:120]
    d_U3 = d_params[120:150]
    d_b0 = d_params[150:180]
    d_w1 = d_params[180:210]
    d_b1 = d_params[210]
    d_h = sigmoid(v*d_v + W_enc*d_wenc + U_1*d_U1 + U_2*d_U2 + U_3*d_U3 + d_b0)
    d_o = np.sum(d_h*d_w1) + d_b1
    return d_o

@jit
def U_1(f_emp, v_max, v_min):
    integrand = f_emp
    mean_integrand = np.mean(integrand)
    return mean_integrand * (v_max-v_min)

@jit
def U_2(f_emp, v, v_max, v_min):
    integrand = np.multiply(f_emp, v)
    mean_integrand = np.mean(integrand)
    return mean_integrand * (v_max-v_min)

@jit
def U_3(f_emp, v, v_max, v_min):
    v_squared = np.multiply(v, v)
    integrand = np.multiply(f_emp, v_squared/2)
    mean_integrand = np.mean(integrand)
    return mean_integrand * (v_max-v_min)

def full_network(params, v, f):
    w_params = params[:90]
    d_params = params[90:300]
    w = w_vect(w_params, v)
    W_enco = W_enc(f, w, max(v), min(v))
    U1 = U_1(f, max(v), min(v))
    U2 = U_2(f, v, max(v), min(v))
    U3 = U_3(f, v, max(v), min(v))
    f_out = decoder_vect(d_params, v, W_enco, U1, U2, U3)
    return f_out

def loss(params, f, v):
    return np.mean((f - full_network(params, v, f))**2)

def nest_momentum(f, v, momentum, lr):
    global params
    global velocity
    gradient = grad_loss(params + momentum*velocity, f, v)
    velocity = momentum*velocity - lr*gradient
    params += velocity

##########
## Main ##
##########

key = random.PRNGKey(0)
params = random.normal(key, shape=(301,)) * 0.3

#-- Setting up the functions and derivatives --#


w_vect = vmap(w, (None, 0))
decoder_vect = vmap(decoder, (None, 0, None, None, None, None))

grad_loss = (grad(loss, 0))

#-- Defining the domain of v, f_emp --#

v_values = np.linspace(0, 4, num=100)

v = []
f = []

for i in range(len(v_values)):
    v.append(v_values[i])
    f.append(2.2*(v_values[i]**2)*np.exp(-(v_values[i]**2)))

v = np.asarray(v)
f = np.asarray(f)

#-- Training parameters --#

#SGD momentum
epochs = 10000
learning_rate = 0.01
momentum = 0.99
velocity = 0.

#-- Training phase --#

for epoch in range(epochs):
    if epoch % 10  == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, f, v)))
    nest_momentum(f, v, momentum, learning_rate)

#-- Plotting and comparison with analytical solution --#

plt.plot(v,2.2*(v**2)*np.exp(-(v**2)), label='exact')
plt.plot(v, full_network(params, v, f), label='approx')
plt.legend()
plt.show()
