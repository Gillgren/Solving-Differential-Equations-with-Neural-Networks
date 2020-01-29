import sys
import jax.numpy as np
from jax import random
from jax import grad
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt

def softplus(x):
    return np.log(1 + np.exp(x))

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def f(params, x):
    w0 = params[:80]
    b0 = params[80:160]
    w1 = params[160:240]
    b1 = params[240]
    h = softplus(x*w0 + b0)
    o = np.sum(h*w1) + b1
    return o

@jit
def loss(params, inputs):
    eq = dfdx_vect(params, inputs) - 2 * inputs * f_vect(params, inputs)
    bc1 = f(params, 0) - 1
    return np.mean(eq**2) + bc1**2

key = random.PRNGKey(0)
params = random.normal(key, shape=(241,))

dfdx = grad(f, 1)
d2fdx2 = grad(dfdx,1)
inputs = np.linspace(-1, 1, num=401)
f_vect = vmap(f, (None, 0))
dfdx_vect = vmap(dfdx, (None, 0))
d2fdx2_vect = vmap(d2fdx2, (None, 0))
grad_loss = jit(grad(loss, 0))

epochs = 20000
learning_rate = 0.0005
momentum = 0.99
velocity = 0.

for epoch in range(epochs):
    if epoch % 100  == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)))
    gradient = grad_loss(params + momentum*velocity, inputs)
    velocity = momentum*velocity - learning_rate*gradient
    params += velocity



plt.plot(inputs, np.exp(inputs**2), label='exact')
plt.plot(inputs, f_vect(params, inputs), label='approx')
plt.legend()
plt.show()
