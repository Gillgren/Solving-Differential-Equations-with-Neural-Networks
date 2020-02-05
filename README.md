# Solving-Differential-Equations-with-Neural-Networks

In supervised learning, the loss function of a Neural Network is usually described by some property including the predicted values of a model and the true values of the model, for example: loss = (y_true - y_predicted)². The loss function is something that we want to minimize to get an optimal model, i.e. loss -> 0. 

Differential equations, like the ODE: y'(x) = y(x), with condition y(x = 0) = 1 can be put on the form y'(x) - y(x) = 0, i.e. the right-hand-side of the equation can be set to zero. Here, y'(x) refers to the derivative of y with respect to x. 

We can approximate y(x) by employing an Artificiall Neural Network so that ANN(x) =~ y(x). Instead of training the Neural Network where the loss function is defined by the difference between a true value and a predicted value, we train it by defining the loss function as the square of the differential equation and the square of the condition, i.e. loss = (ANN'(x) - ANN(x))² + (ANN(x = 0) - 1)². Since we can define the structure of the Neural Network, we can find the derivate of the network with respect to some input x, and we can also find the gradient of the loss function with respect to the internal parameters. 

One of the big advantages of this method is that we can generate an arbitrarily large amount of data, which is one of the key factors of succesfully optimizing any machine learning model. 

This repository contains examples of how some differential equations are solved using this method.
