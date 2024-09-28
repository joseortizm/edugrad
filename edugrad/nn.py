from edugrad.tensor import Tensor
import numpy as np

class MSELoss():
    def forward(self, y_pred, y_true):
        #todo: is istance and isnt istance
        output = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
        return output 
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def __repr__(self):
        return("MSELoss()")

class Linear:
    def __init__(self):
        self.w = Tensor(np.random.randn()) 
        self.b = Tensor(np.random.randn())  

    def forward(self, x):
        return x * self.w + self.b
    
    def __call__(self, x):
        return self.forward(x)
