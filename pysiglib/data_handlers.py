# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from ctypes import c_double, POINTER, cast

import numpy as np
import torch

from .param_checks import check_type, check_type_multiple, check_non_neg, check_dtype, check_dtype_double, ensure_own_contiguous_storage

from .dtypes import DTYPES

from .sig_length import sig_length

class PolyDataHandler:
    def __init__(self, poly1_, poly2_, dimension, degree):

        check_type_multiple(poly1_, "poly1", (np.ndarray, torch.Tensor))
        check_type_multiple(poly2_, "poly2", (np.ndarray, torch.Tensor))
        self.poly1 = ensure_own_contiguous_storage(poly1_, 4)
        self.poly2 = ensure_own_contiguous_storage(poly2_, 4)
        check_dtype_double(self.poly1, "poly1")
        check_dtype_double(self.poly2, "poly2")

        self.poly_len_ = sig_length(dimension, degree)
        if self.poly1.shape[-1] != self.poly_len_:
            raise ValueError("poly1 is of incorrect length. Expected " + str(self.poly_len_) + ", got " + str(self.poly1.shape[-1]))
        if self.poly2.shape[-1] != self.poly_len_:
            raise ValueError("poly2 is of incorrect length. Expected " + str(self.poly_len_) + ", got " + str(self.poly2.shape[-1]))

        if len(self.poly1.shape) == 1:
            self.is_batch = False
            self.batch_size = 1
            self.length_1 = self.poly1.shape[0]
        elif len(self.poly1.shape) == 2:
            self.is_batch = True
            self.batch_size = self.poly1.shape[0]
            self.length_1 = self.poly1.shape[1]
        else:
            raise ValueError("poly1.shape must have length 1 or 2, got length " + str(len(self.poly1.shape)) + " instead.")

        if len(self.poly2.shape) == 1:
            if self.batch_size != 1:
                raise ValueError("poly1, poly2 have different batch sizes")
            self.length_2 = self.poly2.shape[0]
        elif len(self.poly2.shape) == 2:
            if self.batch_size != self.poly2.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = self.poly2.shape[1]
        else:
            raise ValueError("poly2.shape must have length 1 or 2, got length " + str(len(self.poly2.shape)) + " instead.")

        self.result_length = self.batch_size * self.poly_len_

        if isinstance(self.poly1, np.ndarray) and isinstance(self.poly2, np.ndarray):
            self.poly1_ptr = self.poly1.ctypes.data_as(POINTER(c_double))
            self.poly2_ptr = self.poly2.ctypes.data_as(POINTER(c_double))
            if self.is_batch:
                self.out = np.empty(
                    shape=(self.batch_size, self.poly_len_),
                    dtype=np.float64
                )
            else:
                self.out = np.empty(
                    shape=self.poly_len_,
                    dtype=np.float64
                )
            self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(self.poly1, torch.Tensor) and isinstance(self.poly2, torch.Tensor):
            if not (self.poly1.device.type == "cpu" and self.poly2.device.type == "cpu"):
                raise ValueError("Data must be located on the cpu")
            self.poly1_ptr = cast(self.poly1.data_ptr(), POINTER(c_double))
            self.poly2_ptr = cast(self.poly2.data_ptr(), POINTER(c_double))
            if self.is_batch:
                self.out = torch.empty(
                    size=(self.batch_size, self.poly_len_),
                    dtype=torch.float64
                )
            else:
                self.out = torch.empty(
                    size=(self.poly_len_,),
                    dtype=torch.float64
                )
            self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays")

class SigDataHandler:
    def __init__(self, path_, degree, time_aug, lead_lag):
        check_type_multiple(path_, "path",(np.ndarray, torch.Tensor))
        self.path = ensure_own_contiguous_storage(path_, 4)
        check_dtype(self.path, "path")
        check_type(degree, "degree", int)
        check_non_neg(degree, "degree")
        check_type(time_aug, "time_aug", bool)
        check_type(lead_lag, "lead_lag", bool)

        self.degree = degree
        self.time_aug = time_aug
        self.lead_lag = lead_lag

        self.get_dims(self.path)

        if isinstance(self.path, np.ndarray):
            self.init_numpy(self.path)
        elif isinstance(self.path, torch.Tensor):
            if not self.path.device.type == "cpu":
                raise ValueError("Data must be located on the cpu")
            self.init_torch(self.path)

    def init_numpy(self, path):
        self.dtype = str(path.dtype)
        self.data_ptr = path.ctypes.data_as(POINTER(DTYPES[self.dtype]))

        _, dimension_ = self.transformed_dims()
        if self.is_batch:
            self.out = np.empty(
                shape=(self.batch_size, sig_length(dimension_, self.degree)),
                dtype=np.float64
            )
        else:
            self.out = np.empty(
                shape=sig_length(dimension_, self.degree),
                dtype=np.float64
            )
        self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

    def init_torch(self, path):
        self.dtype = str(path.dtype)[6:]
        self.data_ptr = cast(path.data_ptr(), POINTER(DTYPES[self.dtype]))

        _, dimension_ = self.transformed_dims()
        if self.is_batch:
            self.out = torch.empty(
                size=(self.batch_size, sig_length(dimension_, self.degree)),
                dtype=torch.float64
            )
        else:
            self.out = torch.empty(
                size=(sig_length(dimension_, self.degree),),
                dtype=torch.float64
            )
        self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))

    def get_dims(self, path):
        if len(path.shape) == 2:
            self.is_batch = False
            self.length = path.shape[0]
            self.dimension = path.shape[1]


        elif len(path.shape) == 3:
            self.is_batch = True
            self.batch_size = path.shape[0]
            self.length = path.shape[1]
            self.dimension = path.shape[2]

        else:
            raise ValueError("path.shape must have length 2 or 3, got length " + str(len(path.shape)) + " instead.")

    def transformed_dims(self):
        length_ = self.length
        dimension_ = self.dimension
        if self.lead_lag:
            length_ *= 2
            length_ -= 3
            dimension_ *= 2
        if self.time_aug:
            dimension_ += 1
        return length_, dimension_

class SigKernelDataHandler:
    def __init__(self, path1_, path2_, dyadic_order):
        check_type_multiple(path1_, "path1", (np.ndarray, torch.Tensor))
        check_type_multiple(path2_, "path2", (np.ndarray, torch.Tensor))
        self.path1 = ensure_own_contiguous_storage(path1_, 4)
        self.path2 = ensure_own_contiguous_storage(path2_, 4)
        check_dtype(self.path1, "path1")
        check_dtype(self.path2, "path2")

        if isinstance(dyadic_order, tuple) and len(dyadic_order) == 2:
            self.dyadic_order_1 = dyadic_order[0]
            self.dyadic_order_2 = dyadic_order[1]
        elif isinstance(dyadic_order, int):
            self.dyadic_order_1 = dyadic_order
            self.dyadic_order_2 = dyadic_order
        else:
            raise TypeError("dyadic_order must be an integer or a tuple of length 2")

        if self.dyadic_order_1 < 0 or self.dyadic_order_2 < 0:
            raise ValueError("dyadic_order must be a non-negative integer or tuple of non-negative integers")

        if len(self.path1.shape) == 2:
            self.is_batch = False
            self.batch_size = 1
            self.length_1 = self.path1.shape[0]
            self.dimension = self.path1.shape[1]
        elif len(self.path1.shape) == 3:
            self.is_batch = True
            self.batch_size = self.path1.shape[0]
            self.length_1 = self.path1.shape[1]
            self.dimension = self.path1.shape[2]
        else:
            raise ValueError("path1.shape must have length 2 or 3, got length " + str(len(self.path1.shape)) + " instead.")

        if len(self.path2.shape) == 2:
            if self.batch_size != 1:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = self.path2.shape[0]
            if self.dimension != self.path2.shape[1]:
                raise ValueError("path1, path2 have different dimensions")
        elif len(self.path2.shape) == 3:
            if self.batch_size != self.path2.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = self.path2.shape[1]
            if self.dimension != self.path2.shape[2]:
                raise ValueError("path1, path2 have different dimensions")
        else:
            raise ValueError("path2.shape must have length 2 or 3, got length " + str(len(self.path2.shape)) + " instead.")

        if isinstance(self.path1, np.ndarray) and isinstance(self.path2, np.ndarray):
            self.device = "cpu"
            self.out = np.empty(shape=self.batch_size, dtype=np.float64)
            self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(self.path1, torch.Tensor) and isinstance(self.path2, torch.Tensor) and self.path1.device == self.path2.device:
            self.device = self.path1.device.type
            self.out = torch.empty(self.batch_size, dtype=torch.float64, device = self.device)
            self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays on the same device")

        self.torch_path1 = torch.as_tensor(self.path1)  # Avoids data copy
        self.torch_path2 = torch.as_tensor(self.path2)

        if self.is_batch:
            x1 = self.torch_path1[:, 1:, :] - self.torch_path1[:, :-1, :]
            y1 = self.torch_path2[:, 1:, :] - self.torch_path2[:, :-1, :]
        else:
            x1 = (self.torch_path1[1:, :] - self.torch_path1[:-1, :])[None, :, :]
            y1 = (self.torch_path2[1:, :] - self.torch_path2[:-1, :])[None, :, :]

        self.gram = torch.bmm(x1, y1.permute(0, 2, 1))