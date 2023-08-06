from typing import Any, Union, Optional
from juligrad.tensor import Tensor
import juligrad.ops as ops

class Module:
    def forward(self, **kwargs: Any) -> Tensor:
        raise NotImplementedError
    
    def parameters(self, obj = None) -> list[Tensor]: 
        if obj is None: obj = self
        if isinstance(obj, Tensor):
            return [obj]
        
        res = []
        if obj is None:
            return []
        if isinstance(obj, dict):
            for k,v in obj.items(): res += self.parameters(v)
        if isinstance(obj, list):
            for v in obj: res += self.parameters(v)
        if hasattr(obj, '__dict__'):
            return self.parameters(obj.__dict__)
        return res
    
    def __call__(self, *args: Any):
        return self.forward(*args)

Activation = Union[ops.Sigmoid, ops.Identity, ops.ReLU]
  
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
 
class Conv2d(Module):
    def __init__(self, C: int, C_out, kernelSize: int, padding: Optional[int], stride: Optional[int]) -> None:
        super().__init__()
        self.W: Tensor = Tensor.randn(size=(C_out, C, kernelSize, kernelSize), scale=0.01)
        self.b: Tensor = Tensor.zeros(size=(C_out,1))
        self.stride, self.padding = stride, padding
    
    def forward(self, a:Tensor) -> Tensor:
        return ops.Convolute2D().forward(a,self.W, self.b, self.stride, self.padding)

class CategoricalCrossEntropy(Module):
    def forward(self, pred: Tensor, target: Tensor, axis:int=1) -> Tensor:
        return - (target * (pred + Tensor.fromList([[10**-100]]).expand(repeats = pred.shape[0], dim=0).expand(repeats = pred.shape[1], dim=1)).log()).sum(axis=axis)