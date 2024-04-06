import numpy as np
from edugrad.tensor import Tensor

import torch


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



def test_edugrad(x_init, W_init):
    print()
    print("<<Processing Tensor(x_init)>>")
    x = Tensor(x_init) 
    print("x value is:", x)

    print()
    print("<<Processing Tensor(W_init)>>")
    W = Tensor(W_init)
    print("W value is:", W)

    print()
    print("<<Processing x.dot(W)>>")
    out = x.dot(W)
    print("eduGrad out value is:", out)

    print()
    print("<<Processing out.relu()>>")
    outr = out.relu()
    print("eduGrad outr value is:", outr)

    print()
    print("<<Processing outr.logsoftmax()>>")
    outl = outr.logsoftmax()
    print("eduGrad outl is:", outl)

def test_pytorch(x_init, W_init):
    x = torch.tensor(x_init, requires_grad=True)
    W = torch.tensor(W_init, requires_grad=True)

    out = x.matmul(W)
    print("Pytorch out:", out)

    outr = out.relu()
    print("Pytorch outr:", outr)

    outl = torch.nn.functional.log_softmax(outr, dim=1)
    print("Pytorch outl:", outl)

x_init = np.random.randn(1,3).astype(np.float32)
print("value of x_init:", x_init)
W_init = np.random.randn(3,3).astype(np.float32)
print("value of W_init:", W_init)
print()
print("###Testing edugrad###")
test_edugrad(x_init, W_init)
print("######################")
print("###Testing Pytorch###")
test_pytorch(x_init, W_init)
