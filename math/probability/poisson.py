#!/usr/bin/env python3
"""Creates poisson class"""


class Poisson:
    """Creates poisson class"""
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        self.lambtha = float(lambtha)

        if data is None:
            self.data = lambtha
        try:
            lambtha <= 0
        except (ValueError):
            print("lambtha must be a positive value")
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        try:
            len(data) > 1
        except (TypeError):
            print("data must contain multiple values")

        self.lambtha = sum(data) / len(data)
