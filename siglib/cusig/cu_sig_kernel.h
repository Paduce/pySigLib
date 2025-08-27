/* Copyright 2025 Daniil Shmelev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ========================================================================= */

#pragma once
#include "cupch.h"

__device__ double myAtomicAdd(double* address, double val);

inline __device__ void get_a_b(double& a, double& b, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	const double gram_val = gram[idx] * dyadic_frac;
	const double gram_val_2 = gram_val * gram_val * twelth;
	a = 1. + 0.5 * gram_val + gram_val_2;
	b = 1. - gram_val_2;
}

inline __device__ void get_a(double& a, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	double gram_val = gram[idx] * dyadic_frac;
	a = 1. + gram_val * (0.5 + gram_val * twelth);
}

inline __device__ void get_b(double& b, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double twelth = 1. / 12;
	const double gram_val = gram[idx] * dyadic_frac;
	b = 1. - gram_val * gram_val * twelth;
}

inline __device__ void get_a_b_deriv(double& a_deriv, double& b_deriv, const double* gram, uint64_t idx, double dyadic_frac) {
	static const double sixth = 1. / 6;
	const double gram_val = gram[idx] * dyadic_frac;
	b_deriv = -gram_val * sixth * dyadic_frac;
	a_deriv = 0.5 * dyadic_frac - b_deriv;
}

__global__ void goursat_pde(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	const double* gram
);

void sig_kernel_cuda_(
	const double* gram,
	double* out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_,
	bool return_grid
);

__global__ void goursat_pde_deriv(
	double* initial_condition,
	double* a_initial_condition,
	double* b_initial_condition,
	const double* gram,
	const double* deriv,
	const double* k_grid,
	double* out
);

void sig_kernel_backprop_cuda_(
	const double* gram,
	double* out,
	const double* deriv,
	const double* k_grid,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_
);