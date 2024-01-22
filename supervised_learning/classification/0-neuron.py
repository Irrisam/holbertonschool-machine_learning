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
        self.W = np.random.normal(loc=0.0, scale=1.0, size=(1, nx))
        self.b = 0
        self.A = 0
