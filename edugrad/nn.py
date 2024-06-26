import numpy as np
from edugrad.tensor import Tensor

class Sigmoid:
    """
    input: 
     Variable Tensor o Tensor.data, np.array([...])
    """
    def __call__(self, x):
        return 1 / (1 + np.exp(-x.data)) if isinstance(x, Tensor) else 1 / (1 + np.exp(-x))

    def __repr__(self):
        return "Sigmoid()"

class ReLU:
    """
    input: 
     Variable Tensor o Tensor.data, np.array([...])
    """
    def __call__(self, x):
        return np.maximum(0, x.data) if isinstance(x, Tensor) else np.maximum(0, x)

    def __repr__(self):
        return "ReLU()"

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.status_bias = bias 
        # Inicializar los pesos y los sesgos
        self.weights = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def __call__(self, x): 
       """
        Realiza la operación hacia adelante: y = xW + b
        x: entrada de tamaño (batch_size, in_features)
        return: salida de tamaño (batch_size, out_features)
        """
        return x @ self.weights + (self.bias if self.bias is not None else 0)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.status_bias})"





