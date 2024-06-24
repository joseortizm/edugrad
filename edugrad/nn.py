import numpy as np
from edugrad.tensor import Tensor

# Clase Sigmoid
class Sigmoid:
    """
    input: 
     Variable Tensor o Tensor.data, np.array([...])
    """
    def __call__(self, x):
        return 1 / (1 + np.exp(-x.data)) if isinstance(x, Tensor) else 1 / (1 + np.exp(-x))

    def __repr__(self):
        return "Sigmoid()"

# Clase ReLU
class ReLU:
    """
    input: 
     Variable Tensor o Tensor.data, np.array([...])
    """
    def __call__(self, x):
        return np.maximum(0, x.data) if isinstance(x, Tensor) else np.maximum(0, x)

    def __repr__(self):
        return "ReLU()"