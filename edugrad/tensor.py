import numpy as np

class Tensor:
    def __init__(self, data):
        if type(data) != np.ndarray:
            print("Error: constructing tensor with ", data)
            assert(False)
        self.data = data
        self.grad = None
        self._ctx = None
    
    def __str__(self):
        return f"Tensor {self.data} with grad {self.grad}"