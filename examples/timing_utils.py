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

import os.path
import timeit

try:
    import signatory
except:
    signatory = None

# iisignature requires numpy < 2.0
# esig requires numpy >= 2.0
# Will have to run timings separately, installing numpy<2.0 first for iisig
# and then numpy>=2.0 for esig
try:
    import iisignature
except:
    iisignature = None

try:
    import esig
except:
    esig = None

try:
    import sigkernel
except:
    sigkernel = None

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

import pysiglib

def plot_times(
        x,
        ys,
        legend,
        title,
        xlabel,
        ylabel,
        scale,
        filename,
        linestyles = None
):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    if linestyles is None:
        linestyles = ["-"] * len(ys)

    plt.figure(figsize=(4, 3))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for y, ls in zip(ys, linestyles):
        plt.plot(x, y, linestyle = ls)
    plt.legend(legend)
    plt.yscale(scale)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("plots/" + filename + ".png", dpi=300)
    plt.savefig("plots/" + filename + ".pdf", dpi=300)
    plt.show()

def time_iisig_sig(batch_size, length, dimension, degree, device, N, progress_bar = False):
    if device != "cpu":
        raise ValueError("iisignature only supports cpu")

    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        start = timeit.default_timer()
        iisignature.sig(X, degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_sig(batch_size, length, dimension, degree, horner, n_jobs, device, N, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.signature(X, degree, horner = horner, n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


def time_signatory_sig(batch_size, length, dimension, degree, device, N, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension))#.astype("double")
    X = torch.tensor(X, device=device)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        start = timeit.default_timer()
        signatory.signature(X, degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


def time_esig_sig(batch_size, length, dimension, degree, device, N, progress_bar = False):
    if device != "cpu":
        raise ValueError("esig only supports cpu")
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        start = timeit.default_timer()
        # esig cannot handle batches, so loop
        for i in range(batch_size):
            esig.stream2sig(X[i], degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_sigkernel_kernel(batch_size, length, dimension, dyadic_order, device, N, progress_bar = False):
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            start = timeit.default_timer()
            signature_kernel.compute_kernel(X, Y)
            end = timeit.default_timer()
            time_ = end - start
            best_time = min(best_time, time_)
        except:
            continue
    return best_time

def time_sigkernel_kernel_backprop(batch_size, length, dimension, dyadic_order, device, N, progress_bar = False):
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device, requires_grad = True)
    Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)
    derivs = torch.ones(batch_size, device = device)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            K = signature_kernel.compute_kernel(X, Y)
            start = timeit.default_timer()
            K.backward(derivs)
            end = timeit.default_timer()
            time_ = end - start
            best_time = min(best_time, time_)
        except:
            continue
    return best_time


# def timesigkerax(batch_size, length, dimension, dyadic_order, device, N):
#     signature_kernel = SigKernelJax(static_kernel_kind="linear", refinement_factor=1 << dyadic_order)
#
#     X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
#     X = jax.numpy.array(X)
#     Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
#     Y = jax.numpy.array(Y)
#
#     best_time = float('inf')
#     for _ in range(N):
#         if device == "cuda":
#             torch.cuda.empty_cache()
#         start = timeit.default_timer()
#         signature_kernel.kernel_matrix(X, Y)
#         end = timeit.default_timer()
#         time_ = end - start
#         best_time = min(best_time, time_)
#     return best_time


def time_pysiglib_kernel(batch_size, length, dimension, dyadic_order, device, N, n_jobs = -1, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        if device == "cuda":
            torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sig_kernel(X, Y, dyadic_order, n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_kernel_backprop(batch_size, length, dimension, dyadic_order, device, N, n_jobs = -1, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device, requires_grad = True)
    Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)
    derivs = torch.ones(batch_size, device = device)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        if device == "cuda":
            torch.cuda.empty_cache()
        K = pysiglib.torch_api.sig_kernel(X, Y, dyadic_order, n_jobs = n_jobs)
        start = timeit.default_timer()
        K.backward(derivs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_pysiglib_sig_backprop(batch_size, length, dimension, degree, n_jobs, device, N, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    sig = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree)))
    sig_derivs = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree)))

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sig_backprop(X, sig, sig_derivs, degree, n_jobs = n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_iisig_sig_backprop(batch_size, length, dimension, degree, device, N, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    sig_derivs = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree) - 1))

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        iisignature.sigbackprop(sig_derivs, X, degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def time_signatory_sig_backprop(batch_size, length, dimension, degree, device, N, progress_bar = False):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device, requires_grad=True)
    sig_derivs = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree) - 1))
    sig_derivs = torch.tensor(sig_derivs)

    best_time = float('inf')
    loop = tqdm(range(N)) if progress_bar else range(N)
    for _ in loop:
        torch.cuda.empty_cache()
        sig = signatory.signature(X, degree)
        start = timeit.default_timer()
        sig.backward(sig_derivs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time
