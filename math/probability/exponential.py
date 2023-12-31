#!/usr/bin/env python3
"""Exponential class's file"""


class Exponential:
    """Initiatites exponential class"""

    def __init__(self, data=None, lambtha=1.):
        """Initiatites exponential class"""
        π = 3.1415926536
        e = 2.7182818285
        self.data = data
        if data is None:
            self.data = lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """calculates exponential distribution's PDF for period x"""
        if x < 0:
            return 0
        e = 2.7182818285
        return self.lambtha * (e ** -(self.lambtha * x))

    def cdf(self, x):
        """calculates exponential distribution's CDF for period x"""
        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - (e ** -(self.lambtha * x))
