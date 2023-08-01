from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, TypeAlias, NewType, TYPE_CHECKING
import juligrad.ops as ops

DataLike: TypeAlias = NewType('DataLike', np.ndarray) #for now

if TYPE_CHECKING:
    from juligrad.ops import LazyOp


class Tensor():
    def __init__(self, data:DataLike, shape: Optional[Tuple] = None, sourceOp: Optional[ops.Op] = None, requiresGrad: Optional[bool] = True):
        if type(data) is list: return Tensor.fromList(a=data)
        self.data, self.shape, self.sourceOp, self.requiresGrad = data, shape, sourceOp, requiresGrad
        if shape is None: self.shape = data.shape if isinstance(data, np.ndarray) else ()
        self.requiresGrad = requiresGrad
        self.grad = Tensor(data=np.zeros(shape=self.shape, dtype='float64'), requiresGrad=False) if requiresGrad else None
    
    @staticmethod
    def fromList(a: list, **kwargs):
        return Tensor.fromNumpy(np.array(a), **kwargs)
    
    @staticmethod
    def fromNumpy(arr: np.ndarray, **kwargs) -> Tensor:
        return Tensor(data=arr, shape=arr.shape, **kwargs)
    
    @staticmethod
    def fromScalar(scalar: float, **kwargs) -> Tensor:
        return Tensor(data=np.array(scalar), shape=(), **kwargs)
    
    @staticmethod
    def uniform(size: Tuple[int], low: float = 0.0, high: float = 1.0) -> Tensor:
        return Tensor.fromNumpy(np.random.uniform(low = low, high=high, size=size))
    
    @staticmethod
    def randn(size: Tuple[int], loc: float = 0.0, scale: float = 0.0) -> Tensor:
        return Tensor.fromNumpy(np.random.normal(loc=loc, scale=scale, size=size))
    
    @staticmethod
    def zeros(size: Tuple[int]) -> Tensor:
              return Tensor.fromNumpy(np.zeros(shape=size))
    
    def backward(self, grad_out: Optional[Tensor] = None):
        if self.requiresGrad:
            if grad_out is None:
                self.grad = Tensor.fromScalar(1)
            else:
                #print(grad_out.data)
                #print(self.data)
                if not grad_out.shape == self.shape: raise ValueError(f"Shape of grad tensor ({grad_out.shape})  must be same as Tensor: ({self.shape})")
                self.grad.data += grad_out.data

        if self.sourceOp is not None:
            self.sourceOp.backward(self.grad)

    def numpy(self):
        return self.data
    
    def copy(self):
        new =  Tensor(data=self.data, shape=self.shape,sourceOp=self.sourceOp,requiresGrad=self.requiresGrad)
        new.grad.assign(self.grad.data)
        return new
    
    def assign(self, data:DataLike):
        if not data.shape == self.data.shape: raise ValueError(f"Shape mismatch in assign. Current: {self.data.shape}. New: {data.shape}")
        self.data = data

    def zeroGrad(self):
        self.grad = Tensor(data=np.zeros(shape=self.shape, dtype='float64'), requiresGrad=False) if self.requiresGrad else None
    
    def __mul__(self, other): return ops.Mul().forward(self, other)
    def __add__(self, other): return ops.Add().forward(self, other)
    def __sub__(self, other): return ops.Sub().forward(self, other)
    def __sub__(self, other): return ops.Sub().forward(self, other)
    def __matmul__(self, other): return ops.Matmul().forward(self,other)
    def __pow__(self, x): return self * self if x == 2.0 else NotImplementedError
    def __neg__(self): new = self.copy(); new.assign(-self.data); return new
    def __eq__(self, other: Tensor): return Tensor.fromNumpy(self.data == other.data)
    def sigmoid(self): return ops.Sigmoid().forward(self)
    def log(self): return ops.Log().forward(self)
    def expand(self, repeats: int, dim: int = 0): return ops.Expand().forward(self, repeats, dim)
    def sum(self): return ops.Sum().forward(self)
    def round(self, decimals: int): self.data = np.round(self.data, decimals=decimals); return self
    def __str__(self): return self.data.__str__()
    def ___repr__(self): return self.data.__repr__()




    