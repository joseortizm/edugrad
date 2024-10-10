import numpy as np
import matplotlib.pyplot as plt
import random

from edugrad.nn import CrossEntropyLoss
from edugrad.tensor import Tensor

from sklearn.metrics import accuracy_score
from edugrad.optim import SGD

# Download mnist dataset: https://yann.lecun.com/exdb/mnist/
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
y_test = read_labels(test_labels_path) # usado para accuracy

###Graficar un digito###
def plot_number(image, target):
  plt.figure(figsize=(5,5))
  target_string = str(target)
  plt.title(target_string)
  plt.imshow(image.squeeze(), cmap=plt.get_cmap('gray')) #puede recibir imagenes en  uint8 o float32
  plt.axis('off')
  plt.show()
#randomIndex = random.randint(0, 59000)
#digit = X_train[randomIndex].reshape(28,28)
#label = y_train[randomIndex]
#plot_number(digit, label)
###

def norm_reshape(xTrain, xTest):
  train_images = np.asarray(xTrain, dtype=np.float32) / 255.0
  test_images = np.asarray(xTest, dtype=np.float32) / 255.0
  return train_images, test_images 

trainImages, testImages = norm_reshape(X_train, X_test) 

#convertir las etiquetas de clase a formato de codificación one-hot
def to_categorical(y, num_classes=None):
    # Si no se especifica el número de clases, lo deducimos del máximo en y
    if num_classes is None:
        num_classes = np.max(y) + 1  # Sumar 1 porque las clases empiezan en 0
    # Creamos una matriz de ceros con la forma (n_samples, num_classes)
    one_hot = np.zeros((y.shape[0], num_classes))
    # Establecemos los índices correspondientes a 1
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

#Training
y_train_one_hot = to_categorical(y_train)
Wb = Tensor(np.random.randn(784,10))# new initialized weights for gradient descent
batch_size = 32
steps = 20000
criterion = CrossEntropyLoss()
optimizer = SGD([Wb], lr=0.01)
for step in range(steps): 
  ri = np.random.permutation(trainImages.shape[0])[:batch_size]
  Xb, yb = Tensor(trainImages[ri]), Tensor(y_train_one_hot[ri])
  y_predW = Xb.matmul(Wb) # TODO check #s para integrarlo en CrossEntropyLoss
  probs = y_predW.softmax() #
  log_probs = probs.log() #
  zb = yb*log_probs

  outb = zb.reduce_sum(axis = 1) #
  finb = -outb.reduce_sum()  #
  finb.backward()
  if step % 1000 == 0:
    loss = criterion(trainImages, y_train_one_hot, Wb.data) 
    print(f'loss in step {step} is {loss}')
  optimizer.step()

loss = criterion(trainImages,y_train_one_hot,Wb.data)
print(f'loss in final step {step+1} is {loss}')
print(f'accuracy with test data is {accuracy_score(np.argmax(np.matmul(testImages,Wb.data),axis = 1),y_test)*100} %')

#Test
def predict(X, W):
    y_pred = X.matmul(W).softmax()  # Calcula las probabilidades
    return np.argmax(y_pred.data, axis=1)  # Clase con mayor probabilidad

y_test_pred = predict(Tensor(testImages), Wb)

#Plot
num_images = random.sample(range(10000), 10)
plt.figure(figsize=(15, 10))
position = 0
for position, index in enumerate(num_images):
    plt.subplot(2, 5, position + 1)
    plt.imshow(testImages[index].reshape(28, 28), cmap='gray')
    plt.title(f'Predicción: {y_test_pred[index]} \n Real: {y_test[index]}')
    plt.subplots_adjust(bottom=.01, top=.95, hspace= 0.1)
    plt.axis('off')    
plt.show()

