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

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data + other, requires_grad=self.requires_grad)

    #def __sub__(self, other):
    #    if isinstance(other, Tensor):
    #        other = other.data
    #    return Tensor(self.data - other, requires_grad=self.requires_grad)

    #def __mul__(self, other):
    #    if isinstance(other, Tensor):
    #        other = other.data
    #    return Tensor(self.data * other, requires_grad=self.requires_grad)

    #def __truediv__(self, other):
    #    if isinstance(other, Tensor):
    #        other = other.data
    #    return Tensor(self.data / other, requires_grad=self.requires_grad)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(np.dot(self.data, other), requires_grad=self.requires_grad)
    
    #def backward(self, grad=None):
    #    if not self.requires_grad:
    #        raise RuntimeError("This tensor is not marked as requiring gradients")
    #    
    #    if grad is None:
    #        grad = np.ones_like(self.data)
    #    
    #    self.grad += grad

    #def zero_grad(self):
    #    if self.requires_grad:
    #        self.grad.fill(0)