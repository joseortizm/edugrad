import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32) 
        self.requires_grad = requires_grad
        self.grad = None

        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def size(self):
        """
        Retorna el número total de elementos en el Tensor.
        """
        return self.data.size

    def shape(self):
        """
        Retorna la forma (shape) del Tensor.
        """
        return self.data.shape