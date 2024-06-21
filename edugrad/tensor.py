import numpy as np
from functools import partialmethod

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

    def backward(self, allow_fill=True):
        if self._ctx is None:
            return
        if self.grad is None and allow_fill:
            # fill in the first grad with one
            print("Tamaño del tensor:", self.data.size)
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        assert(self.grad is not None)
       
        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t,g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx.arg, g.shape, t.data.shape))
                assert(False)
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)


class Context:
    def __init__(self, arg, *tensors):
        self.arg = arg
        self.parents = tensors
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)


class Function:
    def apply(self, arg, *x):
        ctx = Context(arg, self, *x)
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))
    
class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return input.dot(weight)
   
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        return grad_input, grad_weight
register('dot', Dot)


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register("relu", ReLU)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        def logsumexp(x):
            c = x.max(axis=1)
            return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
        output = input - logsumexp(input).reshape((-1, 1))
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x*y

    @staticmethod
    def backward(ctx, grad_output):
        x,y = ctx.saved_tensors
        return y*grad_output, x*grad_output
register('mul', Mul)


class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * np.ones_like(input)
register('sum', Sum)

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return x+y

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid = 1 / (1 + np.exp(-input))
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid, = ctx.saved_tensors
        grad_input = grad_output * sigmoid * (1 - sigmoid)
        return grad_input
register("sigmoid", Sigmoid)


#class MSELoss():
#    def __call__(self, x, y): 
#        mse = np.mean((x - y) ** 2)
#        return mse

class MSELoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        loss = np.mean((input - target) ** 2)
        return np.array([loss])
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        grad_input = 2 * (input - target) / input.shape[0] #https://stackoverflow.com/questions/65947284/loss-with-custom-backward-function-in-pytorch-exploding-loss-in-simple-mse-exa
        return grad_input, None
    
    #@staticmethod
    #def backward(ctx, grad_output):
    #    input, target = ctx.saved_tensors
    #    grad_input = 2 * (input - target) / input.size
    #    grad_input = grad_input.reshape(input.shape)
    #    return grad_input * grad_output
    #@staticmethod
    #def backward(ctx, grad_output):
    #    input, target = ctx.saved_tensors
    #    grad_input = 2 * (input - target) / input.size
    #    return grad_input.reshape(input.shape) * grad_output
    #@staticmethod
    #def backward(ctx, grad_output):
    #    input, target = ctx.saved_tensors
    #    grad_input = 2 * (input - target) / input.size
    #    return grad_input * grad_output 
    #@staticmethod
    #def backward(ctx, grad_output=1):
    #    input, target = ctx.saved_tensors
    #    grad_input = grad_output * 2 * (input - target) / input.size
    #    grad_target = grad_output * 2 * (target - input) / input.size
    #    return grad_input, grad_target
register("mse_loss", MSELoss) 




