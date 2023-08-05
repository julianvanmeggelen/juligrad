from __future__ import annotations
from typing import Any, Sequence, Union, Optional, TYPE_CHECKING
import numpy as np
import juligrad.tensor as tensor
from juligrad.tensor import Tensor, DataLike

class Op:
    def forward(self, **kwargs: Any) -> Tensor:
        raise NotImplementedError(f"Forward not implemented for {type(self)}")
    
    def backward(self, grad_out: Tensor):
        raise NotImplementedError(f"Backward not implemented for {type(self)}")
    
    def __call__(self, *args: Any) -> Tensor:
        return self.forward(*args)

# *** Binary Ops ***

class Mul(Op):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        if not a.shape == b.shape: raise ValueError(f"Shape mismatch for Op {type(self)}: {a.shape} * {b.shape}")
        return Tensor(data=(a.data*b.data), sourceOp=self)
    
    def backward(self, grad_out: Tensor) -> None:
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

class Matmul(Op):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        if not a.shape[-1] == b.shape[0]: raise ValueError(f"Shape mismatch for Op {type(self)}: {a.shape} @ {b.shape}")
        return Tensor(data=np.matmul(a.data, b.data), sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.b.backward(Tensor(data=np.matmul(self.a.data.T, grad_out.data), requiresGrad=False))
        self.a.backward(Tensor(data=np.matmul(grad_out.data, self.b.data.T), requiresGrad=False))

# *** Unary Ops ***

class Reshape(Op):
    def forward(self, a: Tensor, newShape: tuple[int]) -> Tensor:
        self.a = a
        return Tensor(data=a.data.reshape(newShape), sourceOp = self)

    def backward(self, grad_out: Tensor):
        return Tensor(data=grad_out.data.reshape(self.a.shape), requiresGrad=False)
    
class Flatten(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        return Tensor(data=a.flatten(), sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.reshape(self.a.shape)))
    
class Expand(Op):
    def forward(self, a: Tensor, repeats: int, dim: list[int] = 0) -> Tensor:
        if not all([a.shape[d] == 1 for d in (dim if type(dim) is tuple else (dim,))]): raise ValueError(f"Cannot expand non-singleton dimension {dim} with length {a.shape[dim]}")
        self.dim = dim
        self.a = a
        return Tensor(data=np.repeat(a.data, repeats, axis=dim), sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.data.sum(axis=self.dim, keepdims=True), requiresGrad=False))
    
class Sigmoid(Op):
    def forward(self, a: Tensor) -> Tensor:
        sigx = 1/(1+np.exp(-a.data))
        self.sigx = sigx
        self.a = a
        return Tensor(data=sigx,  sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.data * self.sigx * (1-self.sigx), shape=grad_out.shape, requiresGrad=False))

class Softmax(Op):
    def forward(self, a:Tensor, axis:int = -1):
        """
        a: Tensor with shape (N, d)
        """
        self.a = a
        self.axis=axis
        self.softmaxx: DataLike = np.exp(a.data)/np.sum(np.exp(a.data), axis=axis, keepdims=True)
        return Tensor(data=self.softmaxx, sourceOp = self)

    def backward(self, grad_out:Tensor):
        self.a.backward(Tensor(data = self.softmaxx * (grad_out.data -(grad_out.data * self.softmaxx).sum(axis=self.axis, keepdims=True)), requiresGrad = False))

class ReLU(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        return Tensor(data=np.where(a.data > 0, a.data, 0 ), sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=np.where(self.a.data > 0, grad_out.data, 0 )))
    
class Identity(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        return a
    
    def backward(self, grad_out: Tensor):
        self.a.grad = grad_out

class Log(Op):
    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        return Tensor(data=np.log(a.data),  sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        self.a.backward(Tensor(data=grad_out.data/self.a.data, requiresGrad=False))

# *** Reduce Ops ***

class Sum(Op):
    def forward(self, a: Tensor, axis: Optional[int] = None) -> Tensor:
        self.shape_in = a.shape
        self.a = a
        self.axis=axis
        return Tensor(data=a.data.sum(axis=axis, keepdims=(not axis is None)),  sourceOp=self)
    
    def backward(self, grad_out: Tensor):
        if self.axis is not None:
            self.a.backward(Tensor(data=np.repeat(a=grad_out.data, axis=self.axis, repeats=self.a.shape[self.axis]), requiresGrad=False))
        else:
            self.a.backward(Tensor(data=np.full(shape=self.shape_in, fill_value=grad_out.data), requiresGrad=False))

# class CategoricalCrossEntropy(Op):
#      def forward(self, true: Tensor) -> Tensor:
#         self.shape_in = a.shape
#         self.a = a
#         return Tensor.fromScalar(a.data.sum(),  sourceOp=self)
    
#     def backward(self, grad_out: Tensor):
#         self.a.backward(Tensor(data=np.full(shape=self.shape_in, fill_value=grad_out.data), requiresGrad=False))


# # *** Conv Ops ***
# class Convolute2d(Op):
#     def __init__(self):
#         super().__init__()

#     def forward(self, a: Tensor, kernel: Tensor, bias: Optional[Tensor]=None, stride: Optional[int]=1) -> Tensor:
#         """
#         a: Tensor with shape (N, C_in, H,W)
#         kernel: Tensor with shape (C_out, C_in, Hk,Wk)
#         bias: Tensor with shape (C_out, 1)
#         """
       
#         self.a = a
#         self.kernel = kernel
#         self.bias = bias
#         self.stride: int = stride

#         N, C_in,H,W =a.data.shape
#         kernel_size = kernel.shape[-1]
#         batch_stride, channel_stride, rows_stride, columns_stride = a.data.strides
#         H_out = (H - kernel_size) // stride + 1
#         W_out = (W - kernel_size) // stride + 1

#         #Strided input
#         strided_input = np.lib.stride_tricks.as_strided(
#             a.data,
#             shape=(
#                 N,  
#                 C_in,
#                 H_out,
#                 W_out,
#                 kernel_size,
#                 kernel_size
#             ),
#             strides=(
#                 batch_stride,
#                 channel_stride,
#                 stride*rows_stride,  
#                 stride*columns_stride,
#                 rows_stride,
#                 columns_stride
#             ),
#             writeable=False
#         )
        
#         self.strided_input: DataLike = strided_input

#         feature_map = np.einsum('bchwkt,fckt->bfhw',strided_input, kernel.data) + bias.data[np.newaxis,:,np.newaxis] 

#         return Tensor(data=feature_map, sourceOp = self)
    
#     def backward(self, grad_out: Tensor):
#         """
#         grad_out: Tensor with shape (N, C_out, W_out, H_out)
#         """
#         kernel_rot = np.rot90(self.kernel.data, 2, axes=(2, 3))
#         kernel_size = self.kernel.shape[-1]

#         N, C_in,H,W = self.a.data.shape
#         padding = kernel_size  - 1 
#         #grad_out_pad: DataLike = grad_out.data#
#         dilate = self.stride-1
#         grad_out_dilated: DataLike = grad_out.data
#         if dilate > 0:
#             grad_out_dilated: DataLike = np.insert(grad_out_dilated, range(1, grad_out.shape[2]), 0, axis=2)
#             grad_out_dilated = np.insert(grad_out_dilated, range(1, grad_out.shape[3]), 0, axis=3)
#         grad_out_pad: DataLike = np.pad(grad_out_dilated, pad_width=((0,), (0,), (padding,), (padding,)), mode='constant', constant_values=(0.,))
#         batch_stride, channel_stride, rows_stride, columns_stride = grad_out_pad.strides
#         N, C_out, W_out, H_out = grad_out_pad.shape

#         print(grad_out.shape, padding, grad_out_pad.shape)
#         strided_grad_out =  np.lib.stride_tricks.as_strided(
#             grad_out_pad,
#             shape=(
#                 N,
#                 C_out,
#                 H_out,
#                 W_out,
#                 kernel_size,
#                 kernel_size
#             ),
#             strides=(
#                 batch_stride,
#                 channel_stride,
#                 self.stride * rows_stride,
#                 self.stride * columns_stride,
#                 rows_stride,
#                 columns_stride
#             ),
#             writeable=False
#         )
#         self.bias.backward(Tensor(data=np.sum(grad_out.data, axis=(0, 2, 3))[:,np.newaxis], requiresGrad=False))
#         self.kernel.backward(Tensor(data=np.einsum('bihwkl,bohw->oikl', self.strided_input, grad_out.data), requiresGrad=False))
#         self.a.backward(Tensor(data=np.einsum('bohwkl,oikl->bihw', strided_grad_out, kernel_rot), requiresGrad=False))
#         return NotImplementedError
        

def _getStrides(input:Tensor, output_size:int, kernel_size:int, padding:Optional[int]=0, stride:Optional[int]=1, dilate:Optional[int]=0):
    working_input = input.data
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )


class Convolute2D(Op):
    
    def forward(self, a: Tensor, weight: Tensor, bias: Tensor, stride: Optional[int]=1, padding: Optional[int]=0):
        """
        a: input Tensor of shape (N, C, H, W)
        weight: Tensor of shape (C_out, C, K, K)
        bias: Tensor of shape (C_out, 1)
        """
        self.kernel_size, self.stride, self.padding, self.a, self.weight, self.bias = weight.shape[-1], stride, padding, a, weight, bias

        n, c, h, w = a.shape
        out_h = (h - self.kernel_size + 2 * padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * padding) // self.stride + 1

        self.a_strides: DataLike = _getStrides(a, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)

        out = np.einsum('bihwkl,oikl->bohw', self.a_strides, weight.data)

        # add bias to kernels
        out += bias.data[np.newaxis, :, np.newaxis]

        return Tensor(data=out, sourceOp = self)

    def backward(self, grad_out: Tensor):
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        grad_out_strides = _getStrides(grad_out, self.a.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)
        rot_kern = np.rot90(self.weight.data, 2, axes=(2, 3))

        self.bias.backward(Tensor(data=np.sum(grad_out.data, axis=(0, 2, 3))[:,np.newaxis], requiresGrad=False))
        self.weight.backward(Tensor(data=np.einsum('bihwkl,bohw->oikl', self.a_strides, grad_out.data), requiresGrad=False))
        self.a.backward(Tensor(data=np.einsum('bohwkl,oikl->bihw', grad_out_strides, rot_kern), requiresGrad=False))
