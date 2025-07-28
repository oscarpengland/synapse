import numpy as np
from synapse import Tensor

t0 = Tensor(np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]))

t1 = Tensor(np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]))

a0 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
])

print('t0 * t1 = \n', t0 * t1)
print('t0 @ t1 = \n', t0 @ t1)
print('a0 @ t1 = \n', a0 @ t1)
