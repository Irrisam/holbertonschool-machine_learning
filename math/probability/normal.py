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
