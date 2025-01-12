from edugrad.tensor import Tensor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import edugrad.nn as nn 
import edugrad.optim as optim

import pandas as pd
from sklearn.linear_model import SGDRegressor


def simple_example():
    x = Tensor(2.0)
    y = Tensor(4.0)
    z = x*y
    s = z.sigmoid()
    s.backward()
    print('s:', s)
    print('s.grad:', s.grad)
    print('z.grad:', z.grad)
    print('x.grad:', x.grad)
    print('y.grad:', y.grad)
#simple_example()


def rl_Edugrad_1():
    # Generar datos de regresión
    x, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=0) 
    # TODO: n_samples > 100000: in build_topo visited.add(v) RecursionError: maximum recursion depth exceeded while calling a Python object
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

    # Normalizar datos (Min-Max)
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    xTrain_norm = normalize(xTrain)
    xTest_norm = normalize(xTest)
    yTrain_norm = normalize(yTrain)
    yTest_norm = normalize(yTest)

    # Si quisieras crear tu propia clase para la regresión lineal
    class LinearRegression:
        def __init__(self):
            self.w = Tensor(np.random.randn())  
            self.b = Tensor(np.random.randn())  

        def forward(self, x):
            return x * self.w + self.b

    # Inicializar el modelo y el optimizador
    model = LinearRegression()
    optimizer = optim.SGD([model.w, model.b], lr=0.1)
    epochs = 100 
    losses = []

    #TODO early stopping
    criterion = nn.MSELoss()

    # Entrenamiento
    for epoch in range(epochs):
        epoch += 1

        # Convertir datos a valores de eduGrad 
        inputs = np.array([Tensor(float(i)) for i in xTrain_norm.flatten()])
        labels = np.array([Tensor(float(i)) for i in yTrain_norm.flatten()])

        # Forward pass
        predictions = np.array([model.forward(x) for x in inputs])

        # Calcular pérdida
        #loss = mse_loss(predictions, labels) 
        loss = criterion(predictions, labels)
        losses.append(loss)

        # Backward pass
        loss.backward()

        # Actualizar parámetros
        optimizer.step()

        print(f'Epoch: {epoch} | Loss: {loss.data}')

    # Predicción en el conjunto de prueba
    test_inputs = np.array([Tensor(float(i)) for i in xTest_norm.flatten()])
    test_predictions = np.array([model.forward(x) for x in test_inputs])

    print("model.w.data: ",model.w.data) 
    print("model.b.data:", model.b.data) 
   
    ## Graficar resultados
    
    # Definir los parámetros de la recta
    m = model.w.data  # Pendiente
    b = model.b.data  # Intersección en y

    # Crear valores de x
    x_ = xTest_norm 

    # Calcular los valores de y según la ecuación de la recta
    y_ = m * x_ + b

    # Graficar la recta
    plt.scatter(xTest_norm, yTest_norm, color='orange', label='Datos de prueba')
    plt.plot(x_, y_, color='red', label=f'y = {m}x + {b}')
    plt.title('Graficar una Recta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    ##Graficar Losses
    losses_ = [l.data for l in losses]
    plt.plot(losses_, color='blue', label='Pérdida (Loss)')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

#rl_Edugrad_1()

# example eduGrad with Iris Dataset 
def rl_Edugrad_2():
    # dataset: https://archive.ics.uci.edu/dataset/53/iris

    data = pd.read_csv('../datasets/iris/iris.data', header=None)
    #print(data.head())

    data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
    X = data['PetalLength'].values.reshape(-1, 1) 
    y = data['SepalLength']

    # Inicializar el modelo y el optimizador
    model = nn.Linear()
    optimizer = optim.SGD([model.w, model.b], lr=0.01)
    epochs = 1000 
    losses = []
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch += 1
        # Convertir datos a valores de Edugrad
        inputs = [Tensor(float(i)) for i in X]
        labels =[Tensor(float(i)) for i in y]

        # Forward pass
        predictions = np.array([model.forward(x) for x in inputs])
        
        # Calcular pérdida
        loss = criterion(predictions, labels)
        losses.append(loss)

        # Backward pass
        loss.backward()

        # Actualizar parámetros
        optimizer.step()

        print(f'Epoch: {epoch} | Loss: {loss.data}')

    # Predicciones
    print("model.w.data: ",model.w.data) 
    print("model.b.data:", model.b.data) 

    ## Graficar resultados

    # Definir los parámetros de la recta
    m = model.w.data  # Pendiente
    b = model.b.data  # Intersección en y

    # Crear valores de x_
    x_ = X

    # Calcular los valores de y_ según la ecuación de la recta
    y_ = m * x_ + b    

    # Graficar resultados
    plt.scatter(X, y, color='blue', label='Datos')
    plt.plot(x_, y_, color='red', label='Regresión lineal')
    plt.xlabel('Longitud de los pétalos')
    plt.ylabel('Longitud de los sépalos')
    plt.legend()
    plt.title('Regresión Lineal usando SGDRegressor')
    plt.show()

#rl_Edugrad_2()

# example Iris Dataset with sklearn
def rl_sklearn_2():
    data = pd.read_csv('../datasets/iris/iris.data', header=None)
    #print(data.head())

    data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
    X = data['PetalLength'].values.reshape(-1, 1) 
    y = data['SepalLength']

    # Crear el modelo SGDRegressor
    model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

    # Entrenar el modelo
    model.fit(X, y)

    # Predicciones
    y_pred = model.predict(X)

    plt.scatter(X, y, color='blue', label='Datos')
    plt.plot(X, y_pred, color='red', label='Regresión lineal')
    plt.xlabel('Longitud de los pétalos')
    plt.ylabel('Longitud de los sépalos')
    plt.legend()
    plt.title('Regresión Lineal usando SGDRegressor')
    plt.show()

    print(f'Coeficiente: {model.coef_[0]}')
    print(f'Intersección: {model.intercept_[0]}')

#rl_sklearn_2()
