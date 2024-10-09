
# EduGrad

Deep learning framework for educational purposes designed for beginners in Python and deep learning. Inspired by [PyTorch](https://github.com/pytorch/pytorch), [Micrograd](https://github.com/karpathy/micrograd) and [Tinygrad](https://github.com/tinygrad/tinygrad).


## Examples

### Iris Dataset
```python
# import edugrad ... go to test.py (rl_Edugrad_2 function)
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







