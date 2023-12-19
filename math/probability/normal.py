#!/usr/bin/env python3
"""Normal distribution class"""


class Normal:
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution class with preset settings"""
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data) / len(data)
                var = sum([(ele - self.mean)**2 for ele in data]) / len(data)
                self.stddev = var**0.5
    def z_score(self, x):
        """Calculates the z_score for a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x_score for a given z-score"""
        return self.stddev * z + self.mean

    def pdf(self, x):
        """Calculates the Normal PDF for a given x-value """
        e = 2.7182818285
        π = 3.1415926536
        µ = self.mean
        sdev = self.stddev
        return e**(-0.5 * ((x - µ) / sdev)**2) / (sdev * (2 * π)**0.5)