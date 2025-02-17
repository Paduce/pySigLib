import pysiglib
import iisignature
import numpy as np
import torch
import sigkernel
import timeit

EPSILON = 1e-10

if __name__ == '__main__':
    X = np.random.uniform(size = (10, 100, 5))
    Y = np.random.uniform(size=(10, 100, 5))

    X = torch.tensor(X, device = "cuda")
    Y = torch.tensor(Y, device = "cuda")

    dyadicOrder = 2

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadicOrder)

    start = timeit.default_timer()
    kernel = signature_kernel.compute_kernel(X, Y, 100)
    end = timeit.default_timer()

    print(end - start)
    print(kernel)

    start = timeit.default_timer()
    kernel = pysiglib.sigKernel(X, Y, dyadicOrder)
    end = timeit.default_timer()
    print(end - start)
    print(kernel)