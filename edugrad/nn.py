from edugrad.tensor import Tensor
import numpy as np

class MSELoss():
    def forward(self, y_pred, y_true):
        # TODO: is istance and isnt istance here and others
        output = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred))) / len(y_pred)
        return output 
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class CrossEntropyLoss():
    def __init__(self, reduction='mean'):
        if reduction not in ['sum', 'mean']:
            raise ValueError(f"Tipo invalido: {reduction}. Usa 'sum' o 'mean'.")
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = logits.softmax()
        log_probs = probs.log()
        losses = targets * log_probs  # Shape: (batch_size, num_classes)
        # Reducir por clase para obtener perdidas individuales
        instance_losses = -losses.reduce_sum(axis=1)  # Shape: (batch_size,)
        # Aplicar la reducción seleccionada
        if self.reduction == 'sum':
            return instance_losses.reduce_sum()  
        elif self.reduction == 'mean':
            return instance_losses.reduce_sum() / instance_losses.data.shape[0] 
    
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

class Linear:
    # TODO: init weights and bias with Xavier
    def __init__(self):
        self.w = Tensor(np.random.randn()) 
        self.b = Tensor(np.random.randn())  

    def forward(self, x):
        return x * self.w + self.b
    
    def __call__(self, x):
        return self.forward(x)

