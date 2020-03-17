import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

################################################################################
# Training data
################################################################################

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

#
# Combine data into matrix.
#

x_train = torch.tensor(np.stack([x, t, bc1], axis = 1)).float()
x_bc = x_train.clone()
x_bc[:, 0] = 0.0
x_train = torch.cat([x_train, x_bc], 0)

################################################################################
# Define model and loss function
################################################################################

f = nn.Sequential(nn.Linear(3, 80),
                  nn.Softplus(),
                  nn.Linear(80, 1))

def loss(f, x):
    """
    Compute PDE loss for given pytorch module f (the neural network)
    and training data x.

    Args:
        f: pytroch module used to predict PDE solution
        x: (n, 3) tensor containing the training data
    Returns:
        torch variable representing the loss
    """

    n = x.size()[0] // 2
    x.requires_grad = True
    fx = f(x)
    dfdx, = torch.autograd.grad(fx,
                                x,
                                create_graph=True,
                                retain_graph=True,
                                grad_outputs=torch.ones(fx.shape))
    l_eq = dfdx[:n, 0] + dfdx[:n, 1] - (3 * x[:n, 0]) - x[:n, 1]
    l_bc = fx[n:, 0] - x[n:, 2]
    return (l_eq ** 2).mean() + (l_bc ** 2).mean()


################################################################################
# Define model and loss function
################################################################################

f.train()



optimizer = torch.optim.SGD(f.parameters(), lr = 5e-3, momentum = 0.99)
f.train()

epochs = 1000
for i in range(epochs):

    optimizer.zero_grad()
    l = loss(f, x_train)
    l.backward()
    optimizer.step()

    if (i % 100) == 0:
        print("Epoch {}: ".format(i), l.item())

################################################################################
# Plot results
################################################################################

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
x_test = torch.tensor(np.stack([x, t, bc1], axis = 1)).float()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, t, f(x_test).detach().numpy())
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

