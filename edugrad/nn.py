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

class Softmax:
    """
    input: 
     Variable Tensor o Tensor.data, np.array([...])
    """
    def forward(self, x):
        exps = np.exp(x.data - np.max(x.data))
        equation = exps/np.sum(exps)
        return Tensor(equation)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Softmax()"

class CrossEntropyLoss:
    def forward(self, input, target):
        return (input, target)
    
    def __call__(self, input, target):
        return self.forward(input, target)
    
    def __repr__(self):
        return("CrossEntropyLoss()")

class MSELoss():
    def forward(self, y_pred, y_true):
        n = y_true.size()
        output = np.sum((y_pred - y_true)**2)/n
        return output 
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    def __repr__(self):
        return("MSELoss()")
       
    #def backward(self):
    #    """
    #    Calcula el gradiente de la pérdida MSE con respecto a y_pred.
    #    
    #    Returns:
    #    - gradient (torch.Tensor): Gradiente de la pérdida con respecto a y_pred.
    #    """
    #    num_samples = self.y_pred.size(0)
    #    gradient = (2 / num_samples) * (self.y_pred - self.y_true)
    #    return gradient
    def backward(self):
        """
        Calcula el gradiente de la pérdida MSE con respecto a y_pred.
        
        Returns:
        - gradient (Tensor): Gradiente de la pérdida con respecto a y_pred.
        """
        num_samples = self.y_pred.data.shape[0]
        gradient = (2 / num_samples) * (self.y_pred - self.y_true)
        return gradient




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
#