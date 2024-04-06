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

class Context:
    def __init__(self, arg, *tensors):
        print()
        print("***Go to class Context***")
        self.arg = arg
        print("self.arg<->arg=", arg)
        self.parents = tensors
        print("self.parents<->tensor=", tensors)
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        print("***Go and end to Context in save_for_backward***")
        self.saved_tensors.extend(x)


class Function:
    def apply(self, arg, *x):
        print()
        print("***Go to class Function***")
        print("arg=", arg)
        ctx = Context(arg, self, *x)
        print("ctx=", ctx)
        print("x of class Function=", x)
        print("self.data of class Function=", self.data)
        print("[t.data for t in x]=",[t.data for t in x])   
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        print("ret=", ret)
        ret._ctx = ctx
        return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))
    
class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        print()
        print("***Go to class Dot in forward***")
        ctx.save_for_backward(input, weight)
        print("input is = ", input)
        print("input.dot(",weight,") is = ", input.dot(weight))
        return input.dot(weight)
   
    @staticmethod
    def backward(ctx, grad_output):
        print() 
        print("***Go to class Dot in backward**")
        input, weight = ctx.saved_tensors
        print("input=", input)
        print("weight=", weight)
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.T.dot(input).T
        print("grad_input=", grad_input)
        print("grad_weight=", grad_weight)
        return grad_input, grad_weight

register('dot', Dot)


class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        print() 
        print("***Go to class ReLU in forward***")
        print("input=", input) 
        ctx.save_for_backward(input)
        print("np.maximun(input, 0:)=", np.maximum(input, 0))
        return np.maximum(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        print() 
        print("***Go to class ReLU backward***")
        input, = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input

register("relu", ReLU)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input):
        print() 
        print("***Go to class LogSoftmax in forward***")
        print("ctx=", ctx)
        print("input=", input)
        def logsumexp(x):
            print("x:", x)
            c = x.max(axis=1)
            print("c = x.max(axis=1)=", c)
            print("c.reshape((-1, 1))=", c.reshape((-1, 1)))
            print("x-c.reshape((-1, 1))=", x-c.reshape((-1, 1)))
            print("np.exp(x-c.reshape((-1, 1)))=", np.exp(x-c.reshape((-1, 1))))
            print("np.exp(x-c.reshape((-1, 1))).sum(axis=1)=", np.exp(x-c.reshape((-1, 1))).sum(axis=1))
            print("np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))=", np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1)))
            return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
        output = input - logsumexp(input).reshape((-1, 1))
        print("output", output) 
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        print() 
        print("***Go to brackward in class LogSoftmax***")
        output, = ctx.saved_tensors
        return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)




















