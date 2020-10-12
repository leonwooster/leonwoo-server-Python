import numpy as np
import torch
import torch.nn as nn

points = [10,20,30,45]
npArray = np.array(points)
print(npArray)

m = nn.ReLU()
input = torch.randn(2)
output = m(input)
print(output)


## ignoring a value
a, _, b = (1, 2, 3) # a = 1, b = 3
print(a, b)

a, *_, b = (7, 6, 5, 4, 3, 2, 1)
print(a, b)

for _ in range(5):
    print(_)


#If you have a long digits number, you can separate the group of digits as you like for better understanding.
## different number systems
## you can also check whether they are correct or not by coverting them into integer using "int" method
million = 1_000_000
binary = 0b_0010
octa = 0o_64
hexa = 0x_23_ab

print(million)
print(binary)
print(octa)
print(hexa)    