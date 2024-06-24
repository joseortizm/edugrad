import numpy as np

from edugrad.tensor import Tensor
import edugrad.nn as nn 

import torch



def edugrad_test_tensor(t):
    print("###edugrad_test_tensor###")
    print(t)
    print(t.size())
    print(t.shape()) 

def edugrad_test_funcActivation(t):
    print("###edugrad_test_funcActivation###")
    sigmoid = nn.Sigmoid()
    relu = nn.ReLU()
    output_sigmoid = sigmoid(t)
    print(output_sigmoid)
    output_relu = relu(t) 
    print(output_relu)

def pytorch_test_funcActivation(t):
    print("###pytorch_test_funcActivation###")
    output_sigmoid = torch.sigmoid(t)
    print(output_sigmoid)
    output_relu = torch.relu(t) 
    print(output_relu)

t1 = Tensor([[1, 2, 3], [4, 5, 6]])
t2 = Tensor([-1.0, 0.5, 2.0])
#t3 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

t4_init = np.random.randn(1,3).astype(np.float32)
print("t4_init:", t4_init)

t4_edugrad = Tensor(t4_init)
print("t4_edugrad:", t4_edugrad)

t4_pytorch = torch.tensor(t4_init)
print("t4_pytorch:", t4_pytorch)

edugrad_test_tensor(t4_edugrad)
edugrad_test_funcActivation(t4_edugrad)
pytorch_test_funcActivation(t4_pytorch)















