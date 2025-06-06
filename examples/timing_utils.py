import pysiglib
# import signatory
import iisignature
# import esig

import numpy as np
import timeit
import sigkernel
import jax
from sigkerax.sigkernel import SigKernel as SigKernelJax
from tqdm import tqdm
import torch
import pickle
import matplotlib.pyplot as plt

def plot_times(
        X,
        Ys,
        legend,
        title,
        xlabel,
        ylabel,
        scale,
        filename
):
    plt.figure(figsize=(4, 3))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for Y in Ys:
        plt.plot(X, Y)
    plt.legend(legend)
    plt.yscale(scale)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("plots/" + filename + ".png", dpi=300)
    plt.savefig("plots/" + filename + ".pdf", dpi=300)
    plt.show()

def timeiisig(batchSize, length, dimension, degree, device, N):
    if device != "cpu":
        raise ValueError("iisignature only supports cpu")

    X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")

    best_time = float('inf')
    for i in range(N):
        start = timeit.default_timer()
        iisignature.sig(X, degree)
        end = timeit.default_timer()
        time_ = end - start
        if time_ < best_time:
            best_time = time_
    return best_time

def timepysiglib(batchSize, length, dimension, degree, horner, parallel, device, N):
    X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)

    best_time = float('inf')
    for i in range(N):
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.signature(X, degree, horner = horner, parallel = parallel)
        end = timeit.default_timer()
        time_ = end - start
        if time_ < best_time:
            best_time = time_
    return best_time


# def timesignatory(batchSize, length, dimension, degree, device, N):
#     X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
#     X = torch.tensor(X, device=device)
#
#     best_time = float('inf')
#     for i in range(N):
#         start = timeit.default_timer()
#         signatory.signature(X, degree)
#         end = timeit.default_timer()
#         time_ = end - start
#         if time_ < best_time:
#             best_time = time_
#     return best_time


# def timeesig(batchSize, length, dimension, degree, device, N):
#     if device != "cpu":
#         raise ValueError("esig only supports cpu")
#     X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
#
#     best_time = float('inf')
#     for i in range(N):
#         start = timeit.default_timer()
#         esig.stream2sig(X, degree)
#         end = timeit.default_timer()
#         time_ = end - start
#         if time_ < best_time:
#             best_time = time_
#     return best_time

def timesigkernel(batchSize, length, dimension, dyadicOrder, device, N):
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadicOrder)

    X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    Y = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)

    best_time = float('inf')
    for i in range(N):
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            start = timeit.default_timer()
            signature_kernel.compute_kernel(X, Y)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        except:
            continue
    return best_time


def timesigkerax(batchSize, length, dimension, dyadicOrder, device, N):
    signature_kernel = SigKernelJax(static_kernel_kind="linear", refinement_factor=1 << dyadicOrder)

    X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    X = jax.numpy.array(X)
    Y = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    Y = jax.numpy.array(Y)

    best_time = float('inf')
    for i in range(N):
        if device == "cuda":
            torch.cuda.empty_cache()
        start = timeit.default_timer()
        signature_kernel.kernel_matrix(X, Y)
        end = timeit.default_timer()
        time_ = end - start
        if time_ < best_time:
            best_time = time_
    return best_time


def timepysiglib_kernel(batchSize, length, dimension, dyadicOrder, device, N):
    X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    Y = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)

    best_time = float('inf')
    for i in range(N):
        if device == "cuda":
            torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sigKernel(X, Y, dyadicOrder)
        end = timeit.default_timer()
        time_ = end - start
        if time_ < best_time:
            best_time = time_
    return best_time