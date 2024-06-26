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

#t1 = Tensor([[1, 2, 3], [4, 5, 6]])
#t2 = Tensor([-1.0, 0.5, 2.0])
##t3 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

#t4_init = np.random.randn(1,3).astype(np.float32)
#print("t4_init:", t4_init)
#
#t4_edugrad = Tensor(t4_init)
#print("t4_edugrad:", t4_edugrad)
#
#t4_pytorch = torch.tensor(t4_init)
#print("t4_pytorch:", t4_pytorch)

#edugrad_test_tensor(t4_edugrad)
#edugrad_test_funcActivation(t4_edugrad)
#pytorch_test_funcActivation(t4_pytorch)


def test_nn():
    #print("Edugrad:")
    x = Tensor([[-1.06876432,  1.26828575,  0.67594512], [-1.97010719,  1.5216349,  -0.58012214], [ 0.70961096, -1.73099819, -0.09753269], [-1.07286343,  2.15543405,  0.44647809]])
    #print("x:", x) 
    linear = nn.Linear(3, 2)
    #print("linear:", linear)
    output_linear = linear(x)
    #print("output_linear:", output_linear)

    sigmoid = nn.Sigmoid()
    output_sigmoid = sigmoid(x)
    #print("output_sigmoid:", output_sigmoid)

    relu = nn.ReLU()
    output_relu = relu(x)
    #print("output_relu:", output_relu)

    #print("Pytorch:")
    x_ = torch.tensor([[-1.06876432,  1.26828575,  0.67594512], [-1.97010719,  1.5216349,  -0.58012214], [ 0.70961096, -1.73099819, -0.09753269], [-1.07286343,  2.15543405,  0.44647809]])
    #print("x_:", x_) 
    linear_torch = torch.nn.Linear(3, 2)
    #print("linear_torch:", linear_torch)
    output_linear_torch = linear_torch(x_)
    #print("output_linear_torch:", output_linear_torch)

    sigmoid_torch = torch.nn.Sigmoid()
    output_sigmoid_torch = sigmoid_torch(x_)
    #print("output_sigmoid_torch:", output_sigmoid_torch)

    relu_torch = torch.nn.ReLU()
    output_relu_torch = relu_torch(x_)
    #print("output_relu_torch:", output_relu_torch)

    print("x:", x) 
    print("x_:", x_) 
    print("linear:", linear)
    print("linear_torch:", linear_torch)
    print("sigmoid:",sigmoid)
    print("sigmoid_torch:",sigmoid_torch)
    print("output_sigmoid:", output_sigmoid)
    print("output_sigmoid_torch:", output_sigmoid_torch)
    print("output_relu:", output_relu)
    print("output_relu_torch:", output_relu_torch)

test_nn()





