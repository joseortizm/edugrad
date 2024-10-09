from edugrad.tensor import Tensor
import numpy as np

class MSELoss():
    def forward(self, y_pred, y_true):
        #todo: is istance and isnt istance
        output = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
        return output 
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class CrossEntropyLoss():
  # TODO: input with logits (scores) and targets [2,4,1,4] (true class) similar to pytorch https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  def forward(self, X, Y, W):  
    logits = np.matmul(X, W)
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1)[:, None]
    loss_terms = Y * np.log(softmax_probs)
    loss = np.sum(loss_terms, axis=1) 
    N = X.shape[0]
    average_loss = -(1 / N) * np.sum(loss)

    return average_loss


  def __call__(self, X, Y, W):
    return self.forward(X, Y, W)

class Linear:
    def __init__(self):
        self.w = Tensor(np.random.randn()) 
        self.b = Tensor(np.random.randn())  

    def forward(self, x):
        return x * self.w + self.b
    
    def __call__(self, x):
        return self.forward(x)


