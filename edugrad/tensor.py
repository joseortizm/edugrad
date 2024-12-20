# Inspired by https://github.com/karpathy/micrograd ; https://github.com/joseortizm/micrograd 
import math 
import numpy as np

class Tensor:
    """
    Esta clase representa un valor en una red neuronal. Guarda tanto el resultado de 
    las operaciones realizadas como los gradientes que se usan para ajustar los 
    parámetros del modelo durante el entrenamiento.

    Args:
        data: valor del nodo 
        grad: gradiente del nodo
        _backward: funcion lambda para realizar backward()
        _prev: nodos previos. Si esta vacio imprime 'set()'
        _op: nombre de la operacion que se realizo para crear este nodo
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None 
        self._prev = set(_children) 
        self._op = _op 

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (self, ), 'tanh')
    
        def _backward():
          self.grad = (1 - t**2)*out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        t = 1 / (1 + math.exp(-self.data))
        out = Tensor(t, (self,), 'sigmoid')
        
        def _backward():            
            self.grad = (t*(1-t))*out.grad
        out._backward = _backward
        return out

    def exp(self):
      x = self.data
      out = Tensor(math.exp(x), (self, ), 'exp')
    
      def _backward():
        self.grad += out.data * out.grad  
      out._backward = _backward
      return out

    def shape(self):
        x = self.data
        out = x.shape
        return out 

    def flatten(self):
        x = self.data
        out = Tensor(x.flatten())
        return out 

    def reshape(self,a, b):
        x = self.data
        out = Tensor(x.reshape(a,b))
        return out

    def size(self):
        out = np.size(self.data)
        return out

    def matmul(self, other):
        #todo: x = self.data and check
        out = Tensor(np.matmul(self.data , other.data), (self, other), 'matmul')
        def _backward():
            self.grad += np.dot(out.grad,other.data.T)
            other.grad += np.dot(self.data.T,out.grad)
        out._backward = _backward
        return out

    def softmax(self):
        # todo x = self.data and check
        out =  Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data
        def _backward():
            self.grad += (out.grad - np.reshape(
            np.sum(out.grad * softmax, 1),
            [-1, 1]
              )) * softmax
        out._backward = _backward
        return out

    def log(self):
        #todo x = self.data and check
        out = Tensor(np.log(self.data),(self,),'log')
        def _backward():
            self.grad += out.grad/self.data
        out._backward = _backward
        return out

    def reduce_sum(self,axis = None):
        # todo x= self.data check, ?integrar con CrossEntropyLoss?
        out = Tensor(np.sum(self.data,axis = axis), (self,), 'REDUCE_SUM')
        def _backward():
            output_shape = np.array(self.data.shape)
            output_shape[axis] = 1
            tile_scaling = self.data.shape // output_shape
            grad = np.reshape(out.grad, output_shape)
            self.grad += np.tile(grad, tile_scaling)

        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * float(other)**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


