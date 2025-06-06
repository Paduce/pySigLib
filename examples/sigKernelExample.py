import pysiglib
import numpy as np
import torch
import sigkernel
import timeit

if __name__ == '__main__':

    batchSize = 10
    length = 100
    dim = 5
    dyadicOrder = 0

    X = np.random.uniform(size = (batchSize, length, dim)).astype("double")
    Y = np.random.uniform(size=(batchSize, length, dim)).astype("double")

    X = torch.tensor(X, device = "cuda")
    Y = torch.tensor(Y, device = "cuda")

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadicOrder)

    start = timeit.default_timer()
    kernel = signature_kernel.compute_kernel(X, Y)
    end = timeit.default_timer()

    print(end - start)
    print(kernel)

    start = timeit.default_timer()
    kernel = pysiglib.sigKernel(X, Y, dyadicOrder)
    end = timeit.default_timer()
    print(end - start)
    print(kernel)