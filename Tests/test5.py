import numpy as np


x = [0, 0, 0]

y = [[1],
     [1],
     [1],
     [1]]

z = [[1, 1],
     [1, 1],
     [1, 1],
     [1, 1]]


print(x + y)
print(y + z)
print(z + x)
print(np.matmul((x+y), z))