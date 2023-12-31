from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, TypeAlias, NewType, TYPE_CHECKING
ops = None
DataLike: TypeAlias = NewType('DataLike', np.ndarray) #for now

class Tensor():
    def __init__(self, data:DataLike, shape: Optional[Tuple] = None, sourceOp: Optional[ops.Op] = None, requiresGrad: Optional[bool] = True):
        global ops; import juligrad.ops as ops #Hack, more elegant way of avoiding circular import?
        if type(data) is list: data = np.array(data)
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
    def randn(size: Tuple[int], loc: float = 0.0, scale: float = 1.0) -> Tensor:
        return Tensor.fromNumpy(np.random.normal(loc=loc, scale=scale, size=size))
    
    @staticmethod
    def zeros(size: Tuple[int]) -> Tensor:
              return Tensor.fromNumpy(np.zeros(shape=size))
    
    def backward(self, grad_out: Optional[Tensor] = None):
        if self.requiresGrad:
            if grad_out is None:
                if (not len(self.shape) == 0) or not all([_ == 1 for _ in self.shape]): raise ValueError(f"Backward can only be called on scalar Tensor. (shape = {self.shape})")
                self.grad = Tensor.fromScalar(1)
            else:
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
    def __truediv__(self, other): return ops.Div().forward(self, other)
    def __matmul__(self, other): return ops.Matmul().forward(self,other)
    def __pow__(self, x): return self * self if x == 2.0 else NotImplementedError
    def __neg__(self): return Tensor(data = np.full(self.shape,-1)) * self
    def __eq__(self, other: Tensor): return Tensor.fromNumpy(self.data == other.data)
    def sigmoid(self): return ops.Sigmoid().forward(self)
    def relu(self): return ops.ReLU().forward(self)
    def softmax(self, axis: int = -1): return ops.Softmax().forward(self, axis=axis)
    def log(self): return ops.Log().forward(self)
    def expand(self, repeats: int, dim: int = 0): return ops.Expand().forward(self, repeats, dim)
    def sum(self, axis: int = None): return ops.Sum().forward(self, axis=axis)
    def round(self, decimals: int): self.data = np.round(self.data, decimals=decimals); return self
    def flatten(self, keepBatchDim: bool = True): return ops.Reshape().forward(self, newShape=(self.shape[0], -1) if keepBatchDim else (-1))
    def reshape(self, newShape: tuple[int]): return ops.Reshape().forward(self, newShape=newShape)
    def __str__(self): return f"Tensor({self.data.__str__()})"
    def ___repr__(self): return self.__str__()




    