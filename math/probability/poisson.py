#!/usr/bin/env python3
"""Creates poisson class"""


class Poisson:
    """Creates poisson class"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        self.lambtha = float(lambtha)

        if data is None:
            self.data = lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """instance method that calculates poisson's pmf"""
        k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285

        factorialk = 1

        for i in range(1, k+1):
            factorialk *= i

        return (self.lambtha ** k) * (e ** (-self.lambtha)) / factorialk
