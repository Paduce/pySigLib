import warnings

from .pysiglib import polyLength, signature, sigKernel

try:
    from ._config import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
except ImportError:
    SYSTEM = None
    BUILT_WITH_CUDA = None
    BUILT_WITH_AVX = None
    warnings.warn("Could not import configuration properties from _config.py - package may not have been built correctly.")