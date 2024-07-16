import numpy as np
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

        if requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, gradient=None):
        if self.requires_grad and self._grad_fn:
            if gradient is None:
                gradient = np.ones_like(self.data)
            self.grad = gradient
            self._grad_fn.backward(gradient)

    #def backward(self, grad=None):
    #    if not self.requires_grad:
    #        return
    #    
    #    if grad is None:
    #        grad = np.ones_like(self.data)
    #    
    #    self.grad += grad

    #def zero_grad(self):
    #    if self.grad is not None:
    #        self.grad.fill(0)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data + other, requires_grad=self.requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data * other, requires_grad=self.requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data - other, requires_grad=self.requires_grad)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data ** other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data ** other, requires_grad=self.requires_grad)


    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        else:
            return Tensor(self.data / other, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def shape(self):
        """
        Retorna la forma (shape) del Tensor.
        """
        return self.data.shape

    def size(self):
        """
        Retorna la cantidad de elementos del Tensor.
        """
        return self.data.size
