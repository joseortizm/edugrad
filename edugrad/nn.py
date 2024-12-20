from edugrad.tensor import Tensor
import numpy as np

class MSELoss():
    """
    Esta clase implementa la función de pérdida Error Cuadrático Medio (MSE)

    La fórmula para el MSE es la siguiente:
        MSE =  Sumatoria[(y_pred[i] - y_true[i])^2]*(1/n) donde:
            - y_pred: predicciones del modelo.
            - y_true: valores reales.
            - n: número total de ejemplos.
    Ejemplo:
        >>> criterion = nn.MSELoss()
        >>> loss = criterion(y_prediciones, y_reales)
    """
    def forward(self, y_pred, y_true):
        output = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
        return output 
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class CrossEntropyLoss():
    """
    Esta clase que implementa la función de pérdida Entropía Cruzada

    La fórmula para la entropía cruzada es la siguiente:
        CE = - Σ Σ targets[i, c] * log(softmax(logits)[i, c])  donde:
            - targets[i, c] es la etiqueta real (en formato one-hot) para el ejemplo i y la clase c.
            - softmax(logits[i]) es la probabilidad predicha para el ejemplo i y la clase c.
            - Σ es la suma sobre todos los ejemplos y clases.
    Args:
        reduction (str, opcional): Especifica cómo reducir las pérdidas. Puede ser 'sum' o 'mean'.
                                   'sum' devuelve la suma de todas las pérdidas y 'mean' devuelve el promedio.
                                   El valor predeterminado es 'mean'.
    Ejemplo:
    >>> criterion = CrossEntropyLoss(reduction='sum') 
    >>> loss = criterion(logits, yb)  
    """
    def __init__(self, reduction='mean'):
        if reduction not in ['sum', 'mean']:
            raise ValueError(f"Tipo invalido: {reduction}. Usa 'sum' o 'mean'.")
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = logits.softmax()
        log_probs = probs.log()
        losses = targets * log_probs  # Shape: (batch_size, num_classes)
        # Reducir por clase para obtener perdidas individuales
        instance_losses = -losses.reduce_sum(axis=1)  # Shape: (batch_size,)
        # Aplicar la reducción seleccionada
        if self.reduction == 'sum':
            return instance_losses.reduce_sum()  
        elif self.reduction == 'mean':
            return instance_losses.reduce_sum() / instance_losses.data.shape[0] 
    
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

class Linear:
    # TODO: init weights and bias with Xavier
    """
    Esta clase implementa una capa lineal
    En una capa lineal, los valores de entrada son multiplicados por los pesos y se les suma un sesgo. 
    La fórmula básica es:
        y = x * W + b, donde:
            - x: entrada
            - W: pesos
            - b: sesgo

    Ejemplo:
    >>> model = nn.Linear()
    >>> x = Tensor(np.array([0.5, -1.5])) 
    >>> print(model(x))  # Esto imprimirá la salida de la capa lineal
    """
    def __init__(self):
        self.w = Tensor(np.random.randn()) 
        self.b = Tensor(np.random.randn())  

    def forward(self, x):
        return x * self.w + self.b
    
    def __call__(self, x):
        return self.forward(x)

