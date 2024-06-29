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
        """
        Inicialización de weights con Xavier/Glorot
        """
        self.in_features = in_features
        self.out_features = out_features
        self.status_bias = bias
        limit = np.sqrt(6 / (in_features + out_features))
        self.weights = Tensor(np.random.uniform(-limit, limit, (out_features, in_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    def forward(self, x):
        """
        Realiza la operación hacia adelante: y = xW + b
        todo: proceso en __init__ y forward si bias=False
        """
        return Tensor(x.data @ self.weights.data.T + self.bias.data, requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.status_bias})"
