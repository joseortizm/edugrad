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
y_test = read_labels(test_labels_path) #usado para accuracy

###Check dataset###
#print("X_train[1]:")
#print(X_train[1])
#print("len(X_train[1]):", len(X_train[1])) #784 = 28*28
#print("X_train.shape:", X_train.shape) #a mi ay viene con X_train.shape: (60000, 784) pero en colab no x eso usaron el reshape. ver si yo en verdad necesito en mi funcion norm_...
#print("X_train[1].shape:", X_train[1].shape) #(784,)
#print("X_train.shape:", X_train.shape) #(60000, 784)
#print("len(X_train):", len(X_train)) #60000
#print("type(X_train):", type(X_train)) #numpy.ndarray
#print("X_train.dtype:", X_train.dtype) #uint8
#print("len(y_train):", len(y_train)) #60000
#print("len(X_test):", len(X_test)) #10000
#print("y_test:", y_test[0]) #7
#print("len(y_test):", len(y_test[0])) #10000
##################

###Graficar un digito###
###
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
  #train_images = train_images.reshape(60000,784)
  #test_images = test_images.reshape(10000,784)
  
  #yTrain = keras.utils.to_categorical(yTrain) ###usado en entrenamiento [0 0 0 1 0 0 0 ...] revisar como lograr esto sin keras (antes entender xq lo uso)
  return train_images, test_images 

trainImages, testImages = norm_reshape(X_train, X_test) 
#print(trainImages[0].shape) #(784,)

#convert las etiquetas de clase de un formato de índice a un formato de codificación one-hot
def to_categorical(y, num_classes=None):
    # Si no se especifica el número de clases, lo deducimos del máximo en y
    if num_classes is None:
        num_classes = np.max(y) + 1  # Sumar 1 porque las clases empiezan en 0

    # Creamos una matriz de ceros con la forma (n_samples, num_classes)
    one_hot = np.zeros((y.shape[0], num_classes))

    # Establecemos los índices correspondientes a 1
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot

y_train_one_hot = to_categorical(y_train)
#print(Tensor(y_train_one_hot[0]))

#Analisis del entrenamiento(TODO):
def calculate_loss(X,Y,W):
  return -(1/X.shape[0])*np.sum(np.sum(Y*np.log(np.exp(np.matmul(X,W)) / np.sum(np.exp(np.matmul(X,W)),axis=1)[:, None]),axis = 1))

Wb = Tensor(np.random.randn(784,10))# new initialized weights for gradient descent
batch_size = 32
steps = 20000
for step in range(steps): 
  ri = np.random.permutation(trainImages.shape[0])[:batch_size]
  Xb, yb = Tensor(trainImages[ri]), Tensor(y_train_one_hot[ri])
  y_predW = Xb.matmul(Wb) #todo
  probs = y_predW.softmax() #todo
  log_probs = probs.log() #todo
  zb = yb*log_probs

  outb = zb.reduce_sum(axis = 1) #todo
  finb = -outb.reduce_sum()  #cross entropy loss
  finb.backward()
  if step % 1000 == 0:
    loss = calculate_loss(trainImages,y_train_one_hot,Wb.data) #todo
    print(f'loss in step {step} is {loss}')
  Wb.data = Wb.data- 0.01*Wb.grad
  Wb.grad = 0
loss = calculate_loss(trainImages,y_train_one_hot,Wb.data)
print(f'loss in final step {step+1} is {loss}')










#TODO: class CrossEntropyLoss DONE
#https://machinelearningmastery.com/cross-entropy-for-machine-learning/
#search others urls

def own_CELoss():
  ##Logits:
  Wb = np.random.randn(784,10)# new initialized weights for gradient descent
  #(784, 10) where 
  #Wb = [[w11, w12, w13, ..., w1 10],
  #     [w21, w22, w23, ..., w2 10],
  #     ...,
  #     [w7841, w7842, w7843, ..., w78410]]

  batch_size = 32
  n_imgs = trainImages.shape[0] #60000
  ri = np.random.permutation(n_imgs)[:batch_size] #[first 32 random number between 0 - 59000] 
  '''
  print(ri) #[.....]
  print(ri[0])
  print(trainImages[ri[0]])
  print("suma matmul 1:")
  #print(np.matmul(trainImages[ri[0]], Wb))
  suma1 = sum(np.matmul(trainImages[ri[0]], Wb))
  print(suma1)
  '''

  '''
  out = np.matmul(trainImages[ri], Wb)
  print(out.shape) #32,10
  print(out)#10 elements
  print("suma matmul 2:")
  print(sum(out[0]))
  '''

  '''
  #X = trainImages[ri]  
  #print(X.shape) #(32, 784)
 '''

  #7 oct: continuar con operaciones con 32 imagenes (batch)
  '''
  #trainImages[ri] -> X
  X = trainImages[ri] 
  #Wb -> W
  W = Wb  
  #y_train -> Y
  Y = y_train[:batch_size]


  #Logits:
  logits = np.matmul(X, W)
  #print(logits[0]) [...] #10
  #print(logits.shape) # 32,10

  #Softmax:
  ## expo logits:
  exp_logits = np.exp(logits)
  softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
  #print(softmax_probs.shape) # 32,10
  #print(softmax_probs[0]) #[prob1, prob2....] #10

  ## Calcula la pérdida de entropía cruzada
  Y = Y.reshape(-1,1) # OK
  loss_terms = Y * np.log(softmax_probs)
  #print(loss_terms.shape) #32,10

  ## Suma para cada muestra
  loss = np.sum(loss_terms, axis=1) 
  #print(loss.shape) # 32,

  ## Promedio de la pérdida
  N = X.shape[0]
  average_loss = -(1 / N) * np.sum(loss) 
  print(average_loss)
  '''
  #loss with first 32
  x = trainImages[:batch_size] 
  y = y_train[:batch_size]
  loss = nn.CrossEntropyLoss()
  output = loss(x, y, Wb)
  print("Loss:", output)

#own_CELoss()

