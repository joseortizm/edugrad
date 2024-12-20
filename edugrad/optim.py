class SGD:
    """
    Optimización por Descenso de Gradiente Estocástico (SGD)
    Esta clase implementa el algoritmo de optimización SGD, que actualiza los 
    parámetros restando el gradiente de la función de pérdida, 
    ajustando por la tasa de aprendizaje.

    Args:
        params (iterable): Un iterable de parámetros (usualmente los pesos del modelo) que se optimizarán.
        lr (float, opcional): La tasa de aprendizaje para la actualización de gradientes. El valor por defecto es 0.01.

    Ejemplo:
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> optimizer.step()  # Actualiza los parámetros basándose en los gradientes
    """
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
            param.grad = 0  







