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
    output_relu = relu(output_linear)
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
    output_relu_torch = relu_torch(output_linear_torch)
    #print("output_relu_torch:", output_relu_torch)

    print("x:", x) 
    print("x_:", x_) 
    print("output_linear:", output_linear)
    print("output_linear_torch:", output_linear_torch)
    print("sigmoid:",sigmoid)
    print("sigmoid_torch:",sigmoid_torch)
    print("output_sigmoid:", output_sigmoid)
    print("output_sigmoid_torch:", output_sigmoid_torch)
    print("output_relu:", output_relu)
    print("output_relu_torch:", output_relu_torch)

#test_nn()

def linear1():
    input_data = np.random.randn(128, 20)
    input_tensor = Tensor(input_data)
    linear_layer = nn.Linear(20, 10)
    print("linear_layer:", linear_layer)
    output_tensor = linear_layer(input_tensor)
    print(output_tensor)
    print(output_tensor.shape())

    input_torch = torch.randn(128, 20)
    linear_torch = torch.nn.Linear(20,10)
    pesos = linear_torch.state_dict()
    output_torch = linear_torch(input_torch)
    print(output_torch)
    print(output_torch.shape)

    print("Pesos Edu:")
    print(linear_layer.weights.shape())
    print(linear_layer.weights)
    print(linear_layer.bias.shape())
    print(linear_layer.bias)
    print("Pesos Torch:")
    print("linear_torch['weight'].shape:", pesos['weight'].shape)
    print(pesos['weight']) 
    print("linear_torch['bias'].shape:", pesos['bias'].shape)
    print(pesos['bias'])  



#linear1()

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 


def test_net():

    x, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=0)
    
    x = np.interp(x, (x.min(), x.max()), (10, 20))
    y = np.interp(y, (y.min(), y.max()), (5, 15))
    
    #plt.plot(x, y, '.')
    #plt.xlabel('Years of experience')
    #plt.ylabel('Salary per month ($k)')
    #plt.title('The relationship between experience & salary')
    #plt.show()

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)    
    inputs = Tensor(xTrain)
    #print(inputs.shape()) #(70,1)
    labels = Tensor(yTrain) #(70,)
    #print(labels.shape())
    class Net():
      def __init__(self):
        self.fc1 = nn.Linear(1, 1)
      def forward(self, x):
        out = self.fc1(x)
        return out

    model = Net()
    outputs = model.forward(inputs)
    criterion = nn.MSELoss()
    loss = criterion(outputs, labels)
    print(loss.shape())
    #todo: comparar con Pytorch


test_net()