#!/usr/bin/env python3
"""neuron file from classification project"""
import numpy as np


class Neuron:
    """Class building for neurons"""
    def __init__(self, nx):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.nx = nx
            self.__W = np.random.normal(loc=0.0, scale=1.0, size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """defines a single neuron performing binary classification """
        z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))
        return self.A

    def cost(self, Y, A):
        """caclculates the cost of the model w/ logistic regression"""
        m = Y.shape[1]
        Z = np.multiply(Y, np.log(A))\
            + np.multiply((1 - Y), np.log(1.0000001 - A))
        cost = -(1 / m) * np.sum(Z)
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neurons prediction"""
        predict = self.forward_prop(X)
        tresh_prediction = np.where(predict >= 0.5, 1, 0)
        cost = self.cost(Y, predict)
        return tresh_prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculate one pass of the gradient neuron

        Args:
            X (np.ndarray): shape (nx,m) input data
            Y (np.ndarray): shape (1,m) correct labels for input data
            A (np.ndarray): shape (1,m) activated output of neuron per exemple
            alpha (float, optional): learning rate of gradient. def to 0.05.
        """
        m = Y.shape[1]
        dz = A - Y
        dw = np.dot(dz, X.T) / m
        db = np.sum(dz) / m

        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron

        Args:
            X (np.ndarray): shape (nx,m) input data
            Y (np.ndarray): shape (1,m) correct labels for input data 
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.

        Returns:
            _type_: _description_
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)

            # Gradient descent
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
