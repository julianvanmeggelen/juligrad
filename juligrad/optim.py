from typing import Optional
import numpy as np
from juligrad.tensor import Tensor, DataLike

class Optimizer:
    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zeroGrad(self):
        for param in self.params: param.zeroGrad()

    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float, momentum: Optional[float] = None, weightDecay: Optional[float] = 1.0, nesterov: Optional[bool] = False):
        super().__init__(params, lr)
        self.momentum, self.weightDecay, self.nesterov = momentum, weightDecay, nesterov 
        self.velocity: list[DataLike] = [np.zeros(shape = param.shape) for param in self.params] 

    def step(self):
        for i, param in enumerate(self.params):
            g: DataLike = param.grad.data + self.weightDecay * param.data if self.weightDecay else param.grad.data
            if self.momentum:
                self.velocity[i]  = self.momentum * self.velocity[i] - self.lr * g
                param.data = param.data + self.velocity[i] if not self.nesterov else param.data + self.momentum * self.velocity[i] - self.lr * g
            else:
                param.data = param.data - self.lr * g
