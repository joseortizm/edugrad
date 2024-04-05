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
        print("-Go to class Context->")
        self.arg = arg
        print("self.arg<->arg:", arg)
        self.parents = tensors
        print("self.parents<->tensor:", tensors)
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        print("-Go to Context in save_for_backward->")
        self.saved_tensors.extend(x)


class Function:
    def apply(self, arg, *x):
        print("-Go to class Function->")
        print("arg:", arg)
        ctx = Context(arg, self, *x)
        print("ctx:", ctx)
        print("x of Function:", x)
        print("self.data of Function:", self.data)
        print("[t.data for t in x]",[t.data for t in x])   
        ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
        print("ret:", ret)
        ret._ctx = ctx
        return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))
    
class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
      print("-Go to class Dot in forward->")
      ctx.save_for_backward(input, weight)
      print("input is", input)
      print("input.dot(",weight,") is :", input.dot(weight))
      return input.dot(weight)
   
    @staticmethod
    def backward(ctx, grad_output):
      print("-Go to class Dot in backward->")
      input, weight = ctx.saved_tensors
      print("input:", input)
      print("weight:", weight)
      grad_input = grad_output.dot(weight.T)
      grad_weight = grad_output.T.dot(input).T
      print("grad_input:", grad_input)
      print("grad_weight:", grad_weight)
      return grad_input, grad_weight

register('dot', Dot)








