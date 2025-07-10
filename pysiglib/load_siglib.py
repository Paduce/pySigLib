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

import os
import sys
import platform
import ctypes
from ctypes import c_float, c_double, c_int, c_int32, c_int64, c_uint64, c_bool, POINTER

######################################################
# Figure out how pysiglib was built, in particular
# whether CUDA is being used
######################################################

try:
    from ._config import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
except ImportError as exc:
    raise RuntimeError("Could not import configuration properties from _config.py - package may not have been built correctly.") from exc

if SYSTEM != platform.system():
    raise RuntimeError("System on which pySigLib was built does not match the current system - package may not have been built correctly.")

######################################################
# Load the cpsig and cusig libraries
######################################################

cpsig, cusig = None, None

# winmode = 0 is necessary here
# https://github.com/NVIDIA/warp/issues/24

dir_ = os.path.dirname(sys.modules['pysiglib'].__file__)

if SYSTEM == 'Windows':
    cpsig_path = os.path.join(dir_, 'cpsig.dll')
    cpsig = ctypes.CDLL(cpsig_path, winmode = 0)

    if BUILT_WITH_CUDA:
        cusig_path = os.path.join(dir_, 'cusig.dll')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == "Linux":
    cpsig_path = os.path.join(dir_, 'libcpsig.so')
    cpsig = ctypes.CDLL(cpsig_path, winmode=0)

    if BUILT_WITH_CUDA:
        cusig_path = os.path.join(dir_, 'libcusig.so')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == 'Darwin':
    cpsig_path = os.path.join(dir_, 'libcpsig.dylib')
    cpsig = ctypes.CDLL(cpsig_path)
else:
    raise Exception("Unsupported OS during pysiglib.py")

######################################################
# Set argtypes and restypes for all imported functions
######################################################

cpsig.sig_length.argtypes = (c_uint64, c_uint64)
cpsig.sig_length.restype = c_uint64

cpsig.sig_combine.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64)
cpsig.sig_combine.restype = c_int

cpsig.batch_sig_combine.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_int)
cpsig.batch_sig_combine.restype = c_int

cpsig.signature_int32.argtypes = (POINTER(c_int32), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool)
cpsig.signature_int32.restype = c_int

cpsig.signature_int64.argtypes = (POINTER(c_int64), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool)
cpsig.signature_int64.restype = c_int

cpsig.signature_float.argtypes = (POINTER(c_float), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool)
cpsig.signature_float.restype = c_int

cpsig.signature_double.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool)
cpsig.signature_double.restype = c_int

cpsig.sig_backprop_float.argtypes = (POINTER(c_float), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
cpsig.batch_signature_double.restype = c_int

cpsig.sig_backprop_double.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
cpsig.batch_signature_double.restype = c_int

cpsig.sig_backprop_int32.argtypes = (POINTER(c_int32), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
cpsig.batch_signature_double.restype = c_int

cpsig.sig_backprop_int64.argtypes = (POINTER(c_int64), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_bool, c_bool)
cpsig.batch_signature_double.restype = c_int

cpsig.batch_signature_int32.argtypes = (POINTER(c_int32), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool, c_int)
cpsig.batch_signature_int32.restype = c_int

cpsig.batch_signature_int64.argtypes = (POINTER(c_int64), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool, c_int)
cpsig.batch_signature_int64.restype = c_int

cpsig.batch_signature_float.argtypes = (POINTER(c_float), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool, c_int)
cpsig.batch_signature_float.restype = c_int

cpsig.batch_signature_double.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_bool, c_bool, c_bool, c_int)
cpsig.batch_signature_double.restype = c_int

cpsig.batch_sig_kernel.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_int)
cpsig.batch_sig_kernel.restype = c_int

if BUILT_WITH_CUDA:
    cusig.batch_sig_kernel_cuda.argtypes = (POINTER(c_double), POINTER(c_double), c_uint64, c_uint64, c_uint64, c_uint64, c_uint64, c_uint64)
    cusig.batch_sig_kernel_cuda.restype = c_int