import numpy as np
from synapse import Tensor

t0 = Tensor(np.array([
    [1, 3, 3],
    [3, 1, 3],
    [3, 3, 1],
]))

t1 = Tensor(np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]))

a0 = np.array([
    [1, 3, 3],
    [3, 2, 3],
    [3, 3, 2],
])

print('testing = \n', t0 * t1 + a0 + a0 * t1 / (t0 * 3) + (t1 * 3.4)**1.23 @ t0)
