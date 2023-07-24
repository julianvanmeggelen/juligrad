from __future__ import annotations
from typing import Sequence, Union, TYPE_CHECKING
from juligrad.tensor import LazyTensor, Tensor, DataLike

TensorLike = Union[Tensor,LazyTensor]

class Op:
    def forward(self, **kwargs: TensorLike):
        raise NotImplementedError(f"Forward not implemented for {type(self)}")
    
    def backward(self, **kwargs: TensorLike):
        raise NotImplementedError(f"Backward not implemented for {type(self)}")
    
class LazyOp(Op):
    def __init__(self):
        self.realized = False
        self.realizedDependencies: set[TensorLike] = set()

    def _realizeDependencies(self, *args):
        for arg in args:
            self.realizedDependencies.add(arg)
            if type(arg) is LazyTensor:
                arg.realize()

    def realize(self):
        raise NotImplementedError(f"Realize not implemented for {type(self)}")

class LazyMul(LazyOp):
    def __init__(self):
        super().__init__()

    def forward(self, a: TensorLike, b: TensorLike) -> LazyTensor:
        self.a = a
        self.b = b
        if not a.shape == b.shape: raise ValueError(f"Shape mismatch for LazyOp {type(self)}: {a.shape} * {b.shape}")
        return LazyTensor(sourceOp=self, shape=a.shape)
    
    def backward(self, grad_out: TensorLike) -> TensorLike:
        return self.a
    
    def realize(self) -> DataLike:
        self._realizeDependencies(self.a, self.b)
        self.realized = True
        return self.a.data * self.b.data    

class LazyAdd(LazyOp):
    def __init__(self):
        super().__init__()
        
    def forward(self, a: TensorLike, b: TensorLike) -> LazyTensor:
        self.a = a
        self.b = b
        if not a.shape == b.shape: raise ValueError(f"Shape mismatch for LazyOp {type(self)}: {a.shape} + {b.shape}")
        return LazyTensor(sourceOp=self, shape=a.shape)
    
    def backward(self, grad_out: TensorLike) -> TensorLike:
        return grad_out
    
    def realize(self) -> DataLike:
        self._realizeDependencies(self.a, self.b)
        self.realized = True
        return self.a.data + self.b.data


