import numpy as np
from edugrad.tensor import Tensor

import torch
import torchvision
import torchvision.transforms as  transforms



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



def test_edugrad(x_init, W_init, m_init):
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

    print()
    print("<<Processing outm.mul()>>")
    m = Tensor(m_init)
    outm = outl.mul(m)
    print("eduGrad outm is:", outm)

    print()
    outx = outm.sum()
    print("eduGrad outx is:", outx)

    outx.backward()
    
    print("$$$return eduGrad$$$")
    print("outx.data")
    print(outx.data)
    print("x.grad")
    print(x.grad) 
    print("W.grad")
    print(W.grad) 
    return outx.data, x.grad, W.grad


def test_pytorch(x_init, W_init, m_init):
    x = torch.tensor(x_init, requires_grad=True)
    W = torch.tensor(W_init, requires_grad=True)
    m = torch.tensor(m_init) 

    out = x.matmul(W)
    print("Pytorch out:", out)

    outr = out.relu()
    print("Pytorch outr:", outr)

    outl = torch.nn.functional.log_softmax(outr, dim=1)
    print("Pytorch outl:", outl)

    outm = outl.mul(m)
    print("Pytorch outm:", outm)

    outx = outm.sum()
    print("Pytorch outx:", outx)

    outx.backward()

    print("$$$return Pytorch$$$")
    print("outx.detach().numpy():")
    print(outx.detach().numpy())
    print("x.grad:")
    print(x.grad) 
    print("W.grad:")
    print(W.grad)
    return outx.detach().numpy(), x.grad, W.grad

#x_init = np.random.randn(1,3).astype(np.float32)
#print("value of x_init:", x_init)
#W_init = np.random.randn(3,3).astype(np.float32)
#print("value of W_init:", W_init)
#m_init = np.random.randn(1,3).astype(np.float32)
#print("value of m_init:", m_init)

#print()
#print("###Testing edugrad###")
#test_edugrad(x_init, W_init, m_init)
#print("######################")
#print("###Testing Pytorch###")
#test_pytorch(x_init, W_init, m_init)
#print("######################")
#print("testing...")
#for x,y in zip(test_edugrad(x_init, W_init, m_init), test_pytorch(x_init, W_init, m_init)):
#    print("x:")
#    print(x)
#    print("y:")
#    print(y)
#    print("np.testing.assert_allclose(x, y, atol=1e-6):")
#    np.testing.assert_allclose(x, y, atol=1e-6)



###
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from edugrad.tensor import MSELoss 

#xavier/glorot initialization (weights)
def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)


#l1 = Tensor(layer_init(784, 128))
#l2 = Tensor(layer_init(128, 10))

#x_init = np.random.randn(1,3).astype(np.float32)
#print("value of x_init:", x_init)
#W_init = np.random.randn(3,3).astype(np.float32)
#print("value of W_init:", W_init)
#x = Tensor(x_init)
#print("value of x:", x)
#W = Tensor(W_init)
#print("value of W:", W)
#out = x.dot(W)
#print("value of out:", out)


###
n = 500
features = 2 #features vector
X, y = make_circles(n_samples=n, factor=0.5, noise=0.05) 
#print(X)
#plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue")
#plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red")
#plt.show()

l1 =  Tensor(layer_init(features, 4))
l2 =  Tensor(layer_init(4, 8))
l3 =  Tensor(layer_init(8, 16))
l4 =  Tensor(layer_init(16, 8))
l5 =  Tensor(layer_init(8, 4))
l6 =  Tensor(layer_init(4, 1))

#########
#l_init = layer_init(features, 4)
##b = np.random.rand(1, 4)*2 - 1
#b = layer_init(1,4)

##v1
#X_ = X
#print("X_:")
#print(X_)
#b_ = b 
#print("b_:")
#print(b_)
#z_1 = X_.dot(l_init)+ b_ 
#print("z_1")
#print(z_1)
#
#def sigmoid(x):
#   return 1/(1 + np.exp(-x))
##_x = np.linspace(-5, 5, 100)
##plt.plot(_x, sigmoid(_x))
##plt.show()
#
#a_1 = sigmoid(z_1)
#print("a_1:")
#print(a_1)
#
###edugrad
#l =  Tensor(l_init)
#X = Tensor(X)
#print("X:")
#print(X)
#b = Tensor(b)
#print(b)
#z1 = X.dot(l).add(b)
##z1_ = X.dot(l1)
##z1_ = out_.add(b)
#print("z1:")
#print(z1)
#a1 = z1.sigmoid()
#print("a1:")
#print(a1)
#########

"""
X = Tensor(X)
b1 = layer_init(1,4)
b1 = Tensor(b1)
z1 = X.dot(l1).add(b1)
a1 = z1.sigmoid()

b2 = layer_init(1,8)
b2 = Tensor(b2)
z2 = a1.dot(l2).add(b2) 
a2 = z2.sigmoid()

b3 = layer_init(1,16)
b3 = Tensor(b3)
z3 = a2.dot(l3).add(b3) 
a3 = z3.sigmoid()

b4 = layer_init(1,8)
b4 = Tensor(b4)
z4 = a3.dot(l4).add(b4) 
a4 = z4.sigmoid()

b5 = layer_init(1,4)
b5 = Tensor(b5)
z5 = a4.dot(l5).add(b5) 
a5 = z5.sigmoid()

b6 = layer_init(1,1)
b6 = Tensor(b6)
z6 = a5.dot(l6).add(b6) 
a6 = z6.sigmoid()
print(a6)

y = Tensor(y)
#loss_func = MSELoss()
output = a6.mse_loss(y)
print("output:")
print(output)
"""







def edugrad():
    n = 500
    features = 2 #features vector
    X, y = make_circles(n_samples=n, factor=0.5, noise=0.05) 
    hiddenLayer = Tensor(layer_init(features,3))
    outputLayer = Tensor(layer_init(3, 1))

    X = Tensor(X)
    b1 = layer_init(1,3)
    b1 = Tensor(b1)
    z1 = X.dot(hiddenLayer).add(b1)
    #z1 = X.dot(hiddenLayer)
    print("z1:")
    print(z1)
    a1 = z1.sigmoid()
    print("a1:")
    print(a1)

    b2 = layer_init(1,1)
    b2 = Tensor(b2)
    z2 = a1.dot(outputLayer).add(b2) 
    #z2 = a1.dot(outputLayer)
    print("z2:")
    print(z2)
    a2 = z2.sigmoid()
    print("a2:")
    print(a2)

    y = Tensor(y)
    loss  = a2.mse_loss(y)
    print("loss:")
    print(loss) 

    #a2.backward() #genera error
    #print("$$$return eduGrad$$$")
    #print("a2.data")
    #print(a2.data)
    #print("X.grad")
    #print(X.grad) 
    #print("hiddenLayer.grad")
    #print(hiddenLayer.grad) 


edugrad()

import torch.nn as nn

def pytorch():
    n = 500
    X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1) 
    fc1 = nn.Linear(2, 3, bias=True)  # Capa oculta con 3 neuronas
    fc2 = nn.Linear(3, 1, bias=True)   # Capa de salida con 1 neurona
    x = torch.sigmoid(fc1(X))  # Aplicar sigmoide en la capa oculta
    x = torch.sigmoid(fc2(x)) 
    print("x:")
    print(x)

#pytorch()

#####
# Definición de la red neuronal
class NeuralNetwork:
    def __init__(self):
        # Inicialización de los pesos de las capas
        self.hidden_layer = Tensor(np.random.randn(2, 3))  # Capa oculta con 2 entradas y 3 neuronas
        self.output_layer = Tensor(np.random.randn(3, 1))  # Capa de salida con 3 entradas y 1 neurona
        
    def forward(self, X):
        # Propagación hacia adelante
        out_hidden = X.dot(self.hidden_layer).relu()  # Capa oculta con función de activación ReLU
        out_output = out_hidden.dot(self.output_layer).sigmoid()  # Capa de salida con función de activación sigmoide
        return out_output
    
    def backward(self, X, y):
        # Propagación hacia atrás
        pred = self.forward(X)
        loss = pred.mse_loss(y)  # Cálculo de la pérdida usando la función de pérdida MSELoss
        loss.backward()  # Retropropagación
        
    def update(self, lr):
        # Actualización de los pesos
        self.hidden_layer.data -= lr * self.hidden_layer.grad
        self.output_layer.data -= lr * self.output_layer.grad


"""
# Creación de datos
n = 500
X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
X = Tensor(X)
y = Tensor(y)
# Entrenamiento de la red neuronal
model = NeuralNetwork()
lr = 0.01
epochs = 1000

for epoch in range(epochs):
    # Forward y Backward
    model.backward(X, y)
    
    # Actualización de pesos
    model.update(lr)
    
    # Calcular la pérdida
    loss = model.forward(X).mse_loss(y)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.data[0]}')
"""

    






















