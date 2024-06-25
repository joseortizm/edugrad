import numpy as np
import matplotlib.pyplot as plt
import random

import edugrad.nn as nn 
from edugrad.tensor import Tensor

def read_images(file_path):
    with open(file_path, 'rb') as f:
        # Leer el encabezado del archivo (los primeros 16 bytes)
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        # Leer los datos de imagen
        buffer = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, num_rows * num_cols)
        return data

def read_labels(file_path):
    with open(file_path, 'rb') as f:
        # Leer el encabezado del archivo (los primeros 8 bytes)
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Leer los datos de etiquetas
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

train_images_path = '../datasets/MNIST/raw/train-images-idx3-ubyte'
train_labels_path = '../datasets/MNIST/raw/train-labels-idx1-ubyte'
test_images_path = '../datasets/MNIST/raw/t10k-images-idx3-ubyte'
test_labels_path = '../datasets/MNIST/raw/t10k-labels-idx1-ubyte'

X_train = read_images(train_images_path)
y_train = read_labels(train_labels_path)
X_test = read_images(test_images_path)
y_test = read_labels(test_labels_path)

#print("X_train[1]:")
#print(X_train[1])
#print("len(X_train[1]):", len(X_train[1])) #784 = 28*28
#print("X_train[1].shape:", X_train[1].shape) #(784,)
#print("X_train.shape:", X_train.shape) #(60000, 784)
#print("len(X_train):", len(X_train)) #60000
#print("type(X_train):", type(X_train)) #numpy.ndarray
#print("X_train.dtype:", X_train.dtype) #uint8
#print("len(y_train):", len(y_train)) #60000
#print("len(X_test):", len(X_test)) #10000
#print("len(y_test):", len(y_test)) #10000



###Graficar un digito###
###
def plot_number(image):
  plt.figure(figsize=(5,5))
  plt.imshow(image.squeeze(), cmap=plt.get_cmap('gray')) #puede recibir imagenes en  uint8 o float32
  plt.axis('off')
  plt.show()

#Picture Dataset Train:
#randomIndex = random.randint(0, 59000)
#digit = X_train[randomIndex].reshape(28,28)
#plot_number(digit)
#print(y_train[randomIndex])
###

###Convertir tipo uint8 a float32
###
#x_train = X_train.reshape(60000, -1).astype(np.float32)
#print("type(x_train):", type(x_train)) #numpy.ndarray
#print("x_train.dtype:", x_train.dtype) #float32
#print("x_train.shape:", x_train.shape) #(60000, 784)
#print("x_train[1]:")
#print(x_train[1])
#print(x_train[1].shape) #(784,)
###


x_train = X_train.reshape(60000, -1).astype(np.float32)
x = x_train[1]/255
x = Tensor(x)
linear = nn.Linear(784, 64)
print(linear) #<edugrad.nn.Linear object at 0x1014fbc70> :se podria modificar esta respuesta?
output_linear = linear(x)
print(output_linear) 
print(output_linear.shape()) #64

sigmoid = nn.Sigmoid()
print(sigmoid(output_linear))

relu = nn.ReLU()
print(relu(output_linear))










