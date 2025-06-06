from typing import Union
import numpy as np
import torch
import ctypes
from ctypes import c_float, c_double, c_int32, c_int64, c_bool, POINTER, cast
import os
import sys
import platform

from .errorCodes import errMsg
import warnings

try:
    from ._config import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
except ImportError:
    SYSTEM = None
    BUILT_WITH_CUDA = None
    BUILT_WITH_AVX = None
    raise RuntimeError("Could not import configuration properties from _config.py - package may not have been built correctly.")

if SYSTEM != platform.system():
    raise RuntimeError("System on which pySigLib was built does not match the current system - package may not have been built correctly.")

dir_ = os.path.dirname(sys.modules['pysiglib'].__file__)
print(dir_)

if SYSTEM == 'Windows':
    cpsig_path = os.path.join(dir_, 'cpsig.dll')
    #https://github.com/NVIDIA/warp/issues/24
    cpsig = ctypes.CDLL(cpsig_path, winmode = 0)

    if BUILT_WITH_CUDA:
        cusig_path = os.path.join(dir_, 'cusig.dll')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == 'Darwin':
    cpsig_path = os.path.join(dir_, 'libcpsig.dylib')
    cpsig = ctypes.CDLL(cpsig_path)
else:
    raise Exception("Unsupported OS during pysiglib.py")

def polyLength(dimension : int, degree : int) -> int:
    """
    Returns the length of a truncated signature,

    .. math::
        \\sum_{i=0}^N d^i = \\frac{d^{N+1} - 1}{d - 1},

    where :math:`d` is the dimension of the underlying path and :math:`N`
    is the truncation level of the signature.

    :param dimension: Dimension of the undelying path, :math:`d`
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`
    :type degree: int
    :return: Length of a truncated signature
    :rtype: int
    """
    cpsig.polyLength.argtypes = (c_int64, c_int64)
    cpsig.polyLength.restype = c_int64
    out = cpsig.polyLength(dimension, degree)
    if out == 0:
        raise Exception("Integer overflow encountered in polyLength")
    return out

class sigDataHandler:
    def __init__(self, path, degree, timeAug, leadLag):
        self.degree = degree
        self.timeAug = timeAug
        self.leadLag = leadLag

        self.getDims(path)

        if isinstance(path, np.ndarray):
            self.initNumpy(path)

        if isinstance(path, torch.Tensor):
            self.initTorch(path)

    def initNumpy(self, path):
        if path.dtype == np.int32:
            self.dtype = "int32"
            self.dataPtr = path.ctypes.data_as(POINTER(c_int32))
        elif path.dtype == np.int64:
            self.dtype = "int64"
            self.dataPtr = path.ctypes.data_as(POINTER(c_int64))
        elif path.dtype == np.float32:
            self.dtype = "float32"
            self.dataPtr = path.ctypes.data_as(POINTER(c_float))
        elif path.dtype == np.float64:
            self.dtype = "float64"
            self.dataPtr = path.ctypes.data_as(POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")

        length_, dimension_ = self.transformedDims()
        if self.isBatch:
            self.out = np.empty(shape=(self.batchSize, polyLength(dimension_, self.degree)), dtype=np.float64)
        else:
            self.out = np.empty(shape=polyLength(dimension_, self.degree), dtype=np.float64)
        self.outPtr = self.out.ctypes.data_as(POINTER(c_double))

    def initTorch(self, path):
        if path.dtype == torch.int32:
            self.dtype = "int32"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_int32))
        if path.dtype == torch.int64:
            self.dtype = "int64"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_int64))
        elif path.dtype == torch.float32:
            self.dtype = "float32"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_float))
        elif path.dtype == torch.float64:
            self.dtype = "float64"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")

        length_, dimension_ = self.transformedDims()
        if self.isBatch:
            self.out = torch.empty(size=(self.batchSize, polyLength(dimension_, self.degree)), dtype=torch.float64)
        else:
            self.out = torch.empty(size=(polyLength(dimension_, self.degree),), dtype=torch.float64)
        self.outPtr = cast(self.out.data_ptr(), POINTER(c_double))

    def getDims(self, path):
        if len(path.shape) == 2:
            self.isBatch = False
            self.length = path.shape[0]
            self.dimension = path.shape[1]


        elif len(path.shape) == 3:
            self.isBatch = True
            self.batchSize = path.shape[0]
            self.length = path.shape[1]
            self.dimension = path.shape[2]

        else:
            raise ValueError("path.shape must have length 2 or 3, got length " + str(path.shape) + " instead.")

    def transformedDims(self):
        length_ = self.length
        dimension_ = self.dimension
        if self.leadLag:
            length_ *= 2
            length_ -= 3
            dimension_ *= 2
        if self.timeAug:
            dimension_ += 1
        return (length_, dimension_)


def signature_(data, timeAug = False, leadLag = False, horner = True):
    errCode = 0
    if data.dtype == "int32":
        cpsig.signatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt32.restype = c_int64
        errCode = cpsig.signatureInt32(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "int64":
        cpsig.signatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt64.restype = c_int64
        errCode = cpsig.signatureInt64(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "float32":
        cpsig.signatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureFloat.restype = c_int64
        errCode = cpsig.signatureFloat(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "float64":
        cpsig.signatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureDouble.restype = c_int64
        errCode = cpsig.signatureDouble(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)

    if errCode:
        raise Exception(errMsg[errCode] + " in signature")
    return data.out

def batchSignature_(data, timeAug = False, leadLag = False, horner = True, parallel = True):
    errCode = 0
    if data.dtype == "int32":
        cpsig.batchSignatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt32.restype = c_int64
        errCode = cpsig.batchSignatureInt32(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "int64":
        cpsig.batchSignatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt64.restype = c_int64
        errCode = cpsig.batchSignatureInt64(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "float32":
        cpsig.batchSignatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureFloat.restype = c_int64
        errCode = cpsig.batchSignatureFloat(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "float64":
        cpsig.batchSignatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureDouble.restype = c_int64
        errCode = cpsig.batchSignatureDouble(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)

    if errCode:
        raise Exception(errMsg[errCode] + " in signature")
    return data.out

def signature(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        timeAug : bool = False,
        leadLag : bool = False,
        horner : bool = True,
        parallel : bool = True #TODO: change to n_jobs
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the truncated signature of single path or a batch of paths.
     For a single path :math:`x`, the signature is given by

    .. math::
        S(x)_{[s,t]} := \\left( 1, S(x)^{(1)}_{[s,t]}, \\ldots, S(x)^{(N)}_{[s,t]}\\right) \\in T((\\mathbb{R}^d)),
    .. math::
        S(x)^{(k)}_{[s,t]} := \\int_{s < t_1 < \\cdots < t_k < t} dx_{t_1} \\otimes dx_{t_2} \\otimes \\cdots \\otimes dx_{t_k} \\in \\left(\\mathbb{R}^d\\right)^{\\otimes k}.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape (length, dimension). For a batch of paths, this must
        be of shape (batch size, length, dimension).
    :type path: numpy.ndarray | torch.tensor
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param timeAug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`.
    :type timeAug: bool
    :param leadLag: If set to True, will compute the signatue of the path after applying the lead-lag transformation.
    :type leadLag: bool
    :param horner: If True, will use Horner's algorithm for polynomial multiplication.
    :type horner: bool
    :param parallel: If True, will parallelise the computation.
    :type parallel: bool
    :return: Truncated signature, or a batch of truncated signatures.
    :rtype: numpy.ndarray | torch.tensor
    """
    data = sigDataHandler(path, degree, timeAug, leadLag)
    if data.isBatch:
        return batchSignature_(data, timeAug, leadLag, horner, parallel)
    else:
        return signature_(data, timeAug, leadLag, horner)


class sigKernelDataHandler:
    def __init__(self, path1, path2, dyadicOrder):
        if isinstance(dyadicOrder, tuple) and len(dyadicOrder) == 2:
            self.dyadicOrder1 = dyadicOrder[0]
            self.dyadicOrder2 = dyadicOrder[1]
        elif isinstance(dyadicOrder, int):
            self.dyadicOrder1 = dyadicOrder
            self.dyadicOrder2 = dyadicOrder
        else:
            raise ValueError("dyadicOrder must be an integer or a tuple of length 2")

        if len(path1.shape) == 2:
            self.isBatch = False
            self.batchSize = 1
            self.length1 = path1.shape[0]
            self.dimension = path1.shape[1]
        elif len(path1.shape) == 3:
            self.isBatch = True
            self.batchSize = path1.shape[0]
            self.length1 = path1.shape[1]
            self.dimension = path1.shape[2]
        else:
            raise ValueError("path1.shape must have length 2 or 3, got length " + str(path1.shape) + " instead.")

        if len(path2.shape) == 2:
            if self.batchSize != 1:
                raise ValueError("path1, path2 have different batch sizes")
            self.length2 = path1.shape[0]
            if self.dimension != path1.shape[1]:
                raise ValueError("path1, path2 have different dimensions")
        elif len(path2.shape) == 3:
            if self.batchSize != path1.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length2 = path1.shape[1]
            if self.dimension != path1.shape[2]:
                raise ValueError("path1, path2 have different dimensions")
        else:
            raise ValueError("path2.shape must have length 2 or 3, got length " + str(path2.shape) + " instead.")

        if isinstance(path1, np.ndarray) and isinstance(path2, np.ndarray):
            self.device = "cpu"
            self.out = np.empty(shape=self.batchSize, dtype=np.float64)
            self.outPtr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(path1, torch.Tensor) and isinstance(path2, torch.Tensor) and path1.device == path2.device:
            self.device = path1.device
            self.out = torch.empty(self.batchSize, dtype=torch.float64, device = self.device)
            self.outPtr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays on the same device")

def sigKernel_(data, gram):
    cpsig.batchSigKernel.argtypes = (
    POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    cpsig.batchSigKernel.restype = c_int64

    errCode = cpsig.batchSigKernel(cast(gram.data_ptr(), POINTER(c_double)), data.outPtr, data.batchSize, data.dimension,
                                   data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)

    if errCode:
        raise Exception(errMsg[errCode] + " in sigKernel")
    return data.out

def sigKernelCUDA_(data, gram):
    cusig.batchSigKernelCUDA.argtypes = (
    POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    cusig.batchSigKernelCUDA.restype = c_int64
    errCode = cusig.batchSigKernelCUDA(cast(gram.data_ptr(), POINTER(c_double)), data.outPtr, data.batchSize, data.dimension,
                                   data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)

    if errCode:
        raise Exception(errMsg[errCode] + " in sigKernel")
    return data.out

# @profile
def sigKernel(
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadicOrder : Union[int, tuple] #TODO: add n_jobs
) -> Union[np.ndarray, torch.tensor]: #TODO: add time-aug and lead-lag
    """
    Computes a single signature kernel or a batch of signature kernels.
    The signature kernel of two :math:`d`-dimensional paths :math:`x,y`
    is defined as

    .. math::
        k_{x,y}(s,t) := \\left< S(x)_{[0,s]}, S(y)_{[0, t]} \\right>_{T((\\mathbb{R}^d))}

    where the inner product is defined as

    .. math::
        \\left< A, B \\right> := \\sum_{k=0}^{\\infty} \\left< A_k, B_k \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}}
    .. math::
        \\left< u, v \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}} := \\prod_{i=1}^k \\left< u_i, v_i \\right>_{\\mathbb{R}^d}

    :param path1: The first underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape (length, dimension). For a batch of paths, this must
        be of shape (batch size, length, dimension).
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape (length, dimension). For a batch of paths, this must
        be of shape (batch size, length, dimension).
    :type path2: numpy.ndarray | torch.tensor
    :param dyadicOrder: If set to a positive integer :math:`\\lambda`, will refine the PDE grid by a factor of :math:`2^\\lambda`.
    :type dyadicOrder: int | tuple
    :return: Single signature kernel or batch of signature kernels
    :rtype: numpy.ndarray | torch.tensor
    """
    data = sigKernelDataHandler(path1, path2, dyadicOrder)
    x1 = path1[:, 1:, :] - path1[:, :-1, :]
    y1 = path2[:, 1:, :] - path2[:, :-1, :]
    gram = torch.bmm(x1, y1.permute(0, 2, 1))

    if data.device.type == "cpu":
        return sigKernel_(data, gram)
    else:
        if not BUILT_WITH_CUDA:
            raise RuntimeError("pySigLib was build without CUDA - data must be moved to CPU.")
        return sigKernelCUDA_(data, gram)