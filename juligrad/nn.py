from typing import Any, Union
from juligrad.tensor import Tensor
import juligrad.ops as ops


class Module:
    def forward(self, **kwargs: Any) -> Tensor:
        raise NotImplementedError
    
    def parameters(self, obj = None) -> list[Tensor]: 
        print(type(obj))
        if obj is None: obj = self
        if isinstance(obj, Tensor):
            return [obj]
        
        res = []
        if isinstance(obj, dict):
            for k,v in obj.items(): res += self.parameters(v)
        if isinstance(obj, list):
            for v in obj: res += self.parameters(v)
        if hasattr(obj, '__dict__'):
            return self.parameters(obj.__dict__)
        return res
    
    def __call__(self, *args: Any):
        return self.forward(*args)

Activation = Union[ops.Sigmoid, ops.Identity]
  
class Linear(Module):
    def __init__(self, shapeIn: int, shapeOut: int):
        self.W = Tensor.randn(size=(shapeIn, shapeOut), loc=0.0, scale=(2/(shapeIn + shapeOut))) #glorot
        self.b = Tensor.zeros(size=(1, shapeOut))

    def forward(self, x: Tensor):
        return x @ self.W + self.b.expand(repeats = x.shape[0], dim=0)
    
class MLP(Module):
    def __init__(self, hiddenDim: list[int], activations: Union[list[Activation], Activation]):
        self.linears = [Linear(shapeIn = hiddenDim[i], shapeOut =  hiddenDim[i+1]) for i in range(len(hiddenDim)-1)]
        self.activations = activations

    def forward(self, x: Tensor):
        out = x
        for i, linear in enumerate(self.linears):
            out = linear(out)
            activation = self.activations[i]() if isinstance(self.activations, list) else self.activations()
            out = activation(out)
        return out



    