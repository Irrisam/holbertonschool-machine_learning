#!/usr/bin/env python3
import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(1)
lam = np.random.uniform(0.1, 10.0)
n = np.random.randint(100, 1000)
data = np.random.poisson(lam, n).tolist()
p = Poisson(data)
print(p.pmf(-1))
print(p.pmf(-1.5))