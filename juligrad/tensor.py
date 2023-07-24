from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, TypeAlias, NewType, TYPE_CHECKING
import juligrad.ops as ops

DataLike: TypeAlias = NewType('DataLike', np.ndarray) #for now

if TYPE_CHECKING:
    from juligrad.ops import LazyOp


class TensorLike:
    def __mul__(self, other): return ops.LazyMul().forward(self, other)
    def __add__(self, other): return ops.LazyAdd().forward(self, other)

class Tensor(TensorLike):
    def __init__(self, data:DataLike, shape: Optional[Tuple] = None):
        self.data = data
        self.shape = shape if shape else ()
    
    @classmethod
    def fromNumpy(self, arr: np.ndarray):
        return Tensor(data=arr, shape=arr.shape)

class LazyTensor(TensorLike):
    def __init__(self, sourceOp: LazyOp, shape:Optional[Tuple]):
        self.realized = False
        self.shape = shape 
        self.sourceOp = sourceOp
        self.data = None

    def realize(self):
        self.data = self.sourceOp.realize()
        self.realized = True

    