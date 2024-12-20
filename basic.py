from edugrad.tensor import Tensor

# Forward: calculamos la salida a través de las operaciones
x1 = Tensor(2)
w1 = Tensor(-3)
x1w1 = x1*w1
print('x1w1:', x1w1)

x2 = Tensor(0)
w2 = Tensor(1)

x2w2 = x2*w2
print('x2w2:', x2w2)

x1w1x2w2 = x1w1 + x2w2
print('x1w1x2w2:',x1w1x2w2)

b = Tensor(6.8814)
n = x1w1x2w2 + b
print('n:', n)

o = n.tanh()
print('o:', o)

# Backward: retropropagación para calcular los gradientes
o.backward()

# Ahora imprimimos los gradientes de cada variable
print('o.grad:', o.grad) 
print('n.grad:', n.grad) 
print('b.grad:', b.grad)
print('x1w1x2w2.grad:', x1w1x2w2.grad)
print('x1w1.grad:', x1w1.grad)
print('x2w2.grad:', x2w2.grad)
print('x1.grad:', x1.grad)
print('w1.grad:', w1.grad)
print('x2.grad:', x2.grad)
print('w2.grad:', w2.grad)

# Imprimir caracteristicas de los nodos
#print(o.data)  # Imprime el resultado al obtener el nodo o (es decir el valor de o)
#print(o._prev)  # Imprime el nodo ó los nodos previos de o: en este caso es el nodo n (su valor y gradiente)
#print(o._op)    # Imprime 'tanh' (la operación que creó este nodo)

