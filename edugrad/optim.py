class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            print(param)
            param.data -= self.lr * param.grad
            param.grad = 0  







