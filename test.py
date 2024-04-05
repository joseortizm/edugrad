import numpy as np
from edugrad.tensor import Tensor


#x_init_1 = np.random.randn(1,3) #float64
#x_init_2 = np.random.randn(1,3).astype(np.float32) #float32
#
#print(x_init_1, type(x_init_1))
#print(x_init_2, type(x_init_2))

def checkFloat(number):
    if type(number) == np.float32:
        print("El número es float32")
    elif type(number) == np.float64:
        print("El número es float64")
    else:
        print("El número no es ni float32 ni float64")


#checkFloat(x_init_1[0][0])

x_init = np.random.randn(1,3).astype(np.float32)

print("value of x_init:", x_init)
x = Tensor(x_init) 
print("x value is:", x)

W_init = np.random.randn(3,3).astype(np.float32)
W = Tensor(W_init)
print("W value is:", W)

out = x.dot(W)
print("out value is:", out)

