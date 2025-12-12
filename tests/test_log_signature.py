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

import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10


def level_index(dimension, degree):
    idx = [0]
    for _ in range(1, degree + 2):
        idx.append(idx[-1] * dimension + 1)
    return idx


def tensor_product(sig1, sig2, dimension, degree, level_idx):
    out = np.zeros(level_idx[degree + 1], dtype=np.float64)
    out[0] = sig1[0] * sig2[0]

    for target_level in range(1, degree + 1):
        level_size = level_idx[target_level + 1] - level_idx[target_level]
        level_out = out[level_idx[target_level]: level_idx[target_level] + level_size]
        level_out[...] = 0.0

        for left_level in range(target_level + 1):
            right_level = target_level - left_level
            left_start, left_end = level_idx[left_level], level_idx[left_level + 1]
            right_start, right_end = level_idx[right_level], level_idx[right_level + 1]
            left_block = sig1[left_start:left_end]
            right_block = sig2[right_start:right_end]

            for left_idx, left_val in enumerate(left_block):
                offset = left_idx * right_block.shape[0]
                level_out[offset: offset + right_block.shape[0]] += left_val * right_block
    return out


def exp_from_log(logsig, dimension, degree):
    idx = level_index(dimension, degree)
    sig_len = idx[degree + 1]
    sig = np.zeros(sig_len, dtype=np.float64)
    sig[0] = 1.0

    x = np.zeros(sig_len, dtype=np.float64)
    cursor = 0
    for level in range(1, degree + 1):
        start, end = idx[level], idx[level + 1]
        size = end - start
        x[start:end] = logsig[cursor: cursor + size]
        cursor += size

    pow_curr = x.copy()
    factorial = 1.0
    for k in range(1, degree + 1):
        factorial *= float(k)
        coeff = 1.0 / factorial
        for level in range(1, degree + 1):
            start, end = idx[level], idx[level + 1]
            sig[start:end] += coeff * pow_curr[start:end]
        if k != degree:
            pow_curr = tensor_product(pow_curr, x, dimension, degree, idx)

    return sig


def check_close(a, b):
    a_ = np.array(a)
    b_ = np.array(b)
    assert not np.any(np.abs(a_ - b_) > EPSILON)


def test_log_sig_length_matches_signature():
    for dim in range(1, 5):
        for deg in range(1, 6):
            assert pysiglib.log_sig_length(dim, deg) == pysiglib.sig_length(dim, deg) - 1


def test_log_signature_degree_one_is_increment():
    X = np.array([[0.0, 0.0], [1.0, -2.0]])
    logsig = pysiglib.log_signature(X, 1)
    sig = pysiglib.signature(X, 1)
    check_close(logsig, sig[1:])


def test_log_signature_linear_path_one_dim_has_no_higher_levels():
    X = np.array([[0.0], [1.0], [3.0]])
    for deg in range(1, 5):
        logsig = pysiglib.log_signature(X, deg)
        assert logsig.shape[-1] == pysiglib.log_sig_length(1, deg)
        if deg > 1:
            check_close(logsig[1:], np.zeros_like(logsig[1:]))
        check_close(logsig[0], np.array([3.0]))


def test_log_signature_reconstructs_signature():
    deg = 3
    X = np.random.uniform(size=(10, 3))
    sig = np.array(pysiglib.signature(X, deg))
    logsig = pysiglib.log_signature(X, deg)
    sig_recon = exp_from_log(logsig, 3, deg)
    check_close(sig, sig_recon)


def test_log_sig_combine_matches_concatenation():
    deg = 3
    X1 = np.random.uniform(size=(50, 2))
    X2 = np.random.uniform(size=(50, 2))
    X_concat = np.concatenate((X1, X2), axis=0)
    X2_with_join = np.concatenate((X1[[-1], :], X2), axis=0)

    log1 = pysiglib.log_signature(X1, deg)
    log2 = pysiglib.log_signature(X2_with_join, deg)
    log_combined = pysiglib.log_sig_combine(log1, log2, 2, deg)

    log_concat = pysiglib.log_signature(X_concat, deg)
    check_close(log_concat, log_combined)


def test_log_sig_combine_batch():
    deg = 2
    batch = 8
    X1 = np.random.uniform(size=(batch, 20, 4))
    X2 = np.random.uniform(size=(batch, 20, 4))
    X_concat = np.concatenate((X1, X2), axis=1)
    X2_with_join = np.concatenate((X1[:, [-1], :], X2), axis=1)

    log1 = pysiglib.log_signature(X1, deg)
    log2 = pysiglib.log_signature(X2_with_join, deg)
    log_combined = pysiglib.log_sig_combine(log1, log2, 4, deg)

    log_concat = pysiglib.log_signature(X_concat, deg)
    check_close(log_concat, log_combined)
