from __future__ import annotations
from typing import Any, Sequence, Union, TYPE_CHECKING
import numpy as np
from juligrad.tensor import Tensor, DataLike

class Op:
    def forward(self, **kwargs: Any) -> Tensor:
        raise NotImplementedError(f"Forward not implemented for {type(self)}")
    
    def backward(self, grad_out: Tensor):
        raise NotImplementedError(f"Backward not implemented for {type(self)}")
    
    def __call__(self, *args: Any) -> Tensor:
        return self.forward(*args)

class Mul(Op):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        if not a.shape == b.shape: raise ValueError(f"Shape mismatch for Op {type(self)}: {a.shape} * {b.shape}")
        return Tensor(data=(a.data*b.data), sourceOp=self)
    
    def backward(self, grad_out: Tensor) :
        self.a.backward(Tensor(grad_out.data * self.a.data))
        self.b.backward(Tensor(grad_out.data * self.b.data))

class Add(Op):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        if not a.shape == b.shape:
            self._tryBroadcasting(a,b)
        return Tensor(data = a.data+b.data, sourceOp=self)
    
    def _tryBroadcasting(self, a ,b):
        #TODO: fix
        # if a.shape[0] == 1 and any([_ == 1 for _ in b.shape]):
        #     dim = b.shape.index(1)
        #     a_bc = Expand().forward(a,repeats = b.shape[dim], dim=dim)
        #     return a_bc + b
        
        # elif b.shape[0] == 1 and any([_ == 1 for _ in a.shape]):
        #     dim = a.shape.index(1)
        #     b_bc = Expand().forward(b,repeats = a.shape[dim], dim=dim)
        #     return b_bc + a
        #
        #else:
            raise ValueError(f"Shape mismatch for Op {type(self)}: {a.shape} + {b.shape}")
         
    def backward(self, grad_out: Tensor):
        self.a.backward(grad_out)
        self.b.backward(grad_out)
    
class Sub(Op):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        if not a.shape == b.shape: raise ValueError(f"Shape mismatch for Op {type(self)}: {a.shape} + {b.shape}")
        return Tensor(data = a.data-b.data, sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(grad_out)
        self.b.backward(Tensor(data=-grad_out.data, requiresGrad=False))

class Expand(Op):
    def forward(self, a: Tensor, repeats: int, dim: int = 0) -> Tensor:
        if not a.shape[dim] == 1: raise ValueError(f"Cannot expand non-singleton dimension {dim} with length {a.shape[dim]}")
        self.dim = dim
        self.a = a
        return Tensor(data=np.repeat(a.data, repeats, axis=dim), sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.data.sum(axis=self.dim, keepdims=True), requiresGrad=False))
    
class Matmul(Op):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        if not a.shape[-1] == b.shape[0]: raise ValueError(f"Shape mismatch for Op {type(self)}: {a.shape} @ {b.shape}")
        return Tensor(data=np.matmul(a.data, b.data), sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        # print('grad_out', grad_out)
        # print('grad_out.shape', grad_out.data.shape)
        # print('self.a.shape', self.a.data.shape)
        # print('self.b.shape', self.b.data.shape)

        self.b.backward(Tensor(data=np.matmul(self.a.data.T, grad_out.data), requiresGrad=False))
        self.a.backward(Tensor(data=np.matmul(grad_out.data, self.b.data.T), requiresGrad=False))


class Sigmoid(Op):
    def forward(self, a: Tensor) -> Tensor:
        sigx = 1/(1+np.exp(-a.data))
        self.sigx = sigx
        self.a = a
        return Tensor(data=sigx,  sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.data * self.sigx * (1-self.sigx), shape=grad_out.shape, requiresGrad=False))

class Identity(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        return a
    
    def backward(self, grad_out: Tensor):
        self.a.grad = grad_out

class Sum(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.shape_in = a.shape
        self.a = a
        return Tensor.fromScalar(a.data.sum(),  sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=np.full(shape=self.shape_in, fill_value=grad_out.data), requiresGrad=False))

class Log(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        return Tensor(data=np.log(a.data),  sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.data/self.a.data, requiresGrad=False))




