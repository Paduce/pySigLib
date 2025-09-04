transform_path_cuda
=====================

The functions ``transform_path_cuda_float``, ``transform_path_cuda_double``, ``transform_path_cuda_int32`` and
``transform_path_cuda_int64`` apply path transformations to paths of input type ``float``, ``double``,
``int32`` and ``int64`` respectively. We provide documentation for ``transform_path_cuda_float``
and omit it for the remaining functions as they are identical beyond input type.

.. doxygengroup:: transform_path_cuda_functions
   :content-only:

batch_transform_path_cuda
===========================

The functions ``batch_transform_path_cuda_float``, ``batch_transform_path_cuda_double``, ``batch_transform_path_cuda_int32`` and
``batch_transform_path_cuda_int64`` apply path transformations to batches of paths of input type ``float``, ``double``,
``int32`` and ``int64`` respectively. We provide documentation for ``batch_transform_path_cuda_float``
and omit it for the remaining functions as they are identical beyond input type.

.. doxygengroup:: batch_transform_path_cuda_functions
   :content-only:

transform_path_backprop_cuda
==============================

.. doxygenfunction:: transform_path_backprop_cuda

batch_transform_path_backprop_cuda
===================================

.. doxygenfunction:: batch_transform_path_backprop_cuda