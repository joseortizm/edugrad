
# eduGrad

Deep learning framework for educational purposes designed for beginners in Python and deep learning. Inspired by [PyTorch](https://github.com/pytorch/pytorch), [Micrograd](https://github.com/karpathy/micrograd) and [Tinygrad](https://github.com/tinygrad/tinygrad).

[Official Documentation](https://joseortizm.github.io/edugrad-website/docs/index.html)

## Examples
### Basic operations
```python
from edugrad.tensor import Tensor

x = Tensor(2.0)
y = Tensor(4.0)
z = x*y
s = z.sigmoid()
s.backward()
print('s:', s)
print('s.grad:', s.grad)
print('z.grad:', z.grad)
print('x.grad:', x.grad)
```
### Iris Dataset
```python
# import edugrad ...
# ... code ... go to test.py (rl_Edugrad_2 function)
model = nn.Linear()
optimizer = optim.SGD([model.w, model.b], lr=0.01)
epochs = 1000 
losses = []
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    inputs = [Tensor(float(i)) for i in X]
    labels =[Tensor(float(i)) for i in y]
    
    predictions = np.array([model.forward(x) for x in inputs])
   
    loss = criterion(predictions, labels)
    losses.append(loss)
    loss.backward()

    optimizer.step()

    print(f'Epoch: {epoch} | Loss: {loss.data}')
```

## TODOs
- Implement an example with a Multilayer Perceptron
- Implement the Adam optimizer
- Implement a Transformer architecture with an example of translation
- Implement a Convolutional Neural Networks architecture
- Create documentation for students and beginners
- Anything that can help make eduGrad a better educational tool





