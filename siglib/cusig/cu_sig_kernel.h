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

extern __constant__ uint64_t dimension;
extern __constant__ uint64_t length1;
extern __constant__ uint64_t length2;
extern __constant__ uint64_t dyadic_order_1;
extern __constant__ uint64_t dyadic_order_2;

extern __constant__ double twelth;
extern __constant__ double sixth;
extern __constant__ uint64_t dyadic_length_1;
extern __constant__ uint64_t dyadic_length_2;
extern __constant__ uint64_t main_dyadic_length;
extern __constant__ uint64_t num_anti_diag;
extern __constant__ double dyadic_frac;
extern __constant__ uint64_t gram_length;
extern __constant__ uint64_t grid_length;

__global__ void goursat_pde(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* gram
);

template<bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32(
	double* initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* diagonals,
	double* gram,
	uint64_t iteration,
	int num_threads
) {
	int thread_id = threadIdx.x;

	// Initialise to 1
	for (int i = 0; i < 3; ++i)
		diagonals[i * 33 + thread_id + 1] = 1.;

	// Indices determine the start points of the antidiagonals in memory
	// Instead of swaping memory, we swap indices to avoid memory copy
	int prev_prev_diag_idx = 0;
	int prev_diag_idx = 33;
	int next_diag_idx = 66;

	if (thread_id == 0) {
		diagonals[prev_prev_diag_idx] = initial_condition[0];
		diagonals[prev_diag_idx] = initial_condition[1];
	}

	__syncthreads();

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1ULL;
			else startj = p - dyadic_length_1 + 1;
			if (num_threads + 1 > p) endj = p;
			else endj = num_threads + 1;

			uint64_t j = startj + thread_id;

			if (j < endj) {

				// Make sure correct initial condition is filled in for first thread
				if (thread_id == 0 && p < dyadic_length_1) {
					diagonals[next_diag_idx] = initial_condition[p];
				}

				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_1);
				uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_2);

				double deriv = gram[ii * (length2 - 1) + jj];
				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				diagonals[next_diag_idx + j] = (diagonals[prev_diag_idx + j] + diagonals[prev_diag_idx + j - 1]) * (
					1. + 0.5 * deriv + deriv2) - diagonals[prev_prev_diag_idx + j - 1] * (1. - deriv2);

			}

			// Wait for all threads to finish
			__syncthreads();

			// Overwrite initial condition with result
			// Safe to do since we won't be using initial_condition[p-num_threads] any more
			if (thread_id == 0 && p >= num_threads && p - num_threads < dyadic_length_1)
				initial_condition[p - num_threads] = diagonals[next_diag_idx + num_threads];
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1ULL;
			else startj = p - dyadic_length_2 + 1;
			if (num_threads + 1 > p) endj = p;
			else endj = num_threads + 1;

			uint64_t j = startj + thread_id;

			if (j < endj) {

				// Make sure correct initial condition is filled in for first thread
				if (thread_id == 0 && p < dyadic_length_2) {
					diagonals[next_diag_idx] = initial_condition[p];
				}

				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_2);
				uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_1);

				double deriv = gram[jj * (length2 - 1) + ii];
				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				diagonals[next_diag_idx + j] = (diagonals[prev_diag_idx + j] + diagonals[prev_diag_idx + j - 1]) * (
					1. + 0.5 * deriv + deriv2) - diagonals[prev_prev_diag_idx + j - 1] * (1. - deriv2);

			}

			// Wait for all threads to finish
			__syncthreads();

			// Overwrite initial condition with result
			// Safe to do since we won't be using initial_condition[p-num_threads] any more
			if (thread_id == 0 && p >= num_threads && p - num_threads < dyadic_length_2)
				initial_condition[p - num_threads] = diagonals[next_diag_idx + num_threads];
		}

		// Rotate the diagonals (swap indices, no data copying)
		int temp = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp;

		// Make sure all threads wait for the rotation of diagonals
		__syncthreads();
	}
}

template<bool order> //order is True if dyadic_length_2 <= dyadic_length_1
__device__ void goursat_pde_32_deriv(
	const double deriv_val,
	const uint64_t p0,
	const uint64_t p1,
	double* k_initial_condition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* d_initial_condition,
	double* k_diagonals,
	double* d_diagonals,
	double* gram,
	const uint64_t iteration,
	const int num_threads
) {
	int thread_id = threadIdx.x;

	// Initialise kernel diagonals to 1 and deriv diagonals to 0
	for (int i = 0; i < 3; ++i) {
		k_diagonals[i * 33 + thread_id + 1] = 1.;
		d_diagonals[i * 33 + thread_id + 1] = 0.;
	}

	// Indices determine the start points of the antidiagonals in memory
	// Instead of swaping memory, we swap indices to avoid memory copy
	int prev_prev_diag_idx = 0;
	int prev_diag_idx = 33;
	int next_diag_idx = 66;

	if (thread_id == 0) {
		k_diagonals[prev_prev_diag_idx] = k_initial_condition[0];
		k_diagonals[prev_diag_idx] = k_initial_condition[1];
	}
	if (thread_id == 1) {
		d_diagonals[prev_prev_diag_idx] = d_initial_condition[0];
		d_diagonals[prev_diag_idx] = d_initial_condition[1];
	}

	__syncthreads();

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1ULL;
			else startj = p - dyadic_length_1 + 1;
			if (num_threads + 1 > p) endj = p;
			else endj = num_threads + 1;

			uint64_t j = startj + thread_id;

			if (j < endj) {

				// Make sure correct initial condition is filled in for first thread
				if (thread_id == 0 && p < dyadic_length_1) {
					k_diagonals[next_diag_idx] = k_initial_condition[p];
					d_diagonals[next_diag_idx] = d_initial_condition[p];
				}

				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_1);
				uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_2);

				double gram_val = gram[ii * (length2 - 1) + jj];
				gram_val *= dyadic_frac;
				double gram_val_2 = gram_val * gram_val * twelth;

				double a = 1. + 0.5 * gram_val + gram_val_2;
				double b = 1. - gram_val_2;

				//Each run of this function is effectively only one PDE run, despite us working with
				//both k(x,y) and its derivative, dk(x,y).

				//The trick is that most of the time we only care about one of k and dk:
				//	- If ii < p0 or jj < p1, then dk = 0, so we only need to update k
				//	- If ii > p0 and jj > p1, then the update rule for dk only depends on dk, so we only need to update dk
				//	- If ii == p0 and jj == p1, then we need both, but this is a small sub-grid of size dyadic_order_1 * dyadic_order_2

				if (ii < p0 || jj < p1) {
					k_diagonals[next_diag_idx + j] =
						(k_diagonals[prev_diag_idx + j] + k_diagonals[prev_diag_idx + j - 1]) * a
						- k_diagonals[prev_prev_diag_idx + j - 1] * b;
				}
				else if (ii == p0 && jj == p1) {
					k_diagonals[next_diag_idx + j] =
						(k_diagonals[prev_diag_idx + j] + k_diagonals[prev_diag_idx + j - 1]) * a
						- k_diagonals[prev_prev_diag_idx + j - 1] * b;

					double b_deriv = -gram_val * sixth * dyadic_frac;
					double a_deriv = 0.5 * dyadic_frac - b_deriv;

					d_diagonals[next_diag_idx + j] =
						(d_diagonals[prev_diag_idx + j] + d_diagonals[prev_diag_idx + j - 1]) * a
						+ (k_diagonals[prev_diag_idx + j] + k_diagonals[prev_diag_idx + j - 1]) * a_deriv
						- d_diagonals[prev_prev_diag_idx + j - 1] * b
						- k_diagonals[prev_prev_diag_idx + j - 1] * b_deriv;
				}
				else {
					d_diagonals[next_diag_idx + j] =
						(d_diagonals[prev_diag_idx + j] + d_diagonals[prev_diag_idx + j - 1]) * a
						- d_diagonals[prev_prev_diag_idx + j - 1] * b;
				}

			}

			// Wait for all threads to finish
			__syncthreads();

			// Overwrite initial condition with result
			// Safe to do since we won't be using initial_condition[p-num_threads] any more
			if (thread_id == 0 && p >= num_threads && p - num_threads < dyadic_length_1) {
				k_initial_condition[p - num_threads] = k_diagonals[next_diag_idx + num_threads];
				d_initial_condition[p - num_threads] = d_diagonals[next_diag_idx + num_threads];
			}
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1ULL;
			else startj = p - dyadic_length_2 + 1;
			if (num_threads + 1 > p) endj = p;
			else endj = num_threads + 1;

			uint64_t j = startj + thread_id;

			if (j < endj) {

				// Make sure correct initial condition is filled in for first thread
				if (thread_id == 0 && p < dyadic_length_2) {
					k_diagonals[next_diag_idx] = k_initial_condition[p];
					d_diagonals[next_diag_idx] = d_initial_condition[p];
				}

				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_2);
				uint64_t jj = ((j + iteration * 32 - 1) >> dyadic_order_1);

				double gram_val = gram[jj * (length2 - 1) + ii];

				gram_val *= dyadic_frac;
				double gram_val_2 = gram_val * gram_val * twelth;

				double a = 1. + 0.5 * gram_val + gram_val_2;
				double b = 1. - gram_val_2;

				//Each run of this function is effectively only one PDE run, despite us working with
				//both k(x,y) and its derivative, dk(x,y).

				//The trick is that most of the time we only care about one of k and dk:
				//	- If ii < p0 or jj < p1, then dk = 0, so we only need to update k
				//	- If ii > p0 and jj > p1, then the update rule for dk only depends on dk, so we only need to update dk
				//	- If ii == p0 and jj == p1, then we need both, but this is a small sub-grid of size dyadic_order_1 * dyadic_order_2

				if (ii < p1 || jj < p0) {
					k_diagonals[next_diag_idx + j] =
						(k_diagonals[prev_diag_idx + j] + k_diagonals[prev_diag_idx + j - 1]) * a
						- k_diagonals[prev_prev_diag_idx + j - 1] * b;
				}
				else if (ii == p1 && jj == p0) {

					k_diagonals[next_diag_idx + j] =
						(k_diagonals[prev_diag_idx + j] + k_diagonals[prev_diag_idx + j - 1]) * a
						- k_diagonals[prev_prev_diag_idx + j - 1] * b;

					double b_deriv = -gram_val * sixth * dyadic_frac;
					double a_deriv = 0.5 * dyadic_frac - b_deriv;

					d_diagonals[next_diag_idx + j] =
						(d_diagonals[prev_diag_idx + j] + d_diagonals[prev_diag_idx + j - 1]) * a
						+ (k_diagonals[prev_diag_idx + j] + k_diagonals[prev_diag_idx + j - 1]) * a_deriv
						- d_diagonals[prev_prev_diag_idx + j - 1] * b
						- k_diagonals[prev_prev_diag_idx + j - 1] * b_deriv;
				}
				else {
					d_diagonals[next_diag_idx + j] =
						(d_diagonals[prev_diag_idx + j] + d_diagonals[prev_diag_idx + j - 1]) * a
						- d_diagonals[prev_prev_diag_idx + j - 1] * b;
				}

			}

			// Wait for all threads to finish
			__syncthreads();

			// Overwrite initial condition with result
			// Safe to do since we won't be using initial_condition[p-num_threads] any more
			if (thread_id == 0 && p >= num_threads && p - num_threads < dyadic_length_2) {
				k_initial_condition[p - num_threads] = k_diagonals[next_diag_idx + num_threads];
				d_initial_condition[p - num_threads] = d_diagonals[next_diag_idx + num_threads];
			}
		}

		// Rotate the diagonals (swap indices, no data copying)
		int temp = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp;

		// Make sure all threads wait for the rotation of diagonals
		__syncthreads();
	}

	// If we've reached the end of the grid, then update the last point to be the derivative:
	// dF / dx{p0,p1} = (dF / dk) * (dk / dx{p0,p1})
	if (thread_id == 0 && num_threads < 32)
		d_initial_condition[main_dyadic_length - 1] *= deriv_val;
}

void sig_kernel_cuda_(
	double* gram,
	double* out,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_
);

__global__ void goursat_pde_deriv(
	double* k_initial_condition,
	double* d_initial_condition,
	double* gram,
	double* deriv
);

void sig_kernel_backprop_cuda_(
	double* gram,
	double* out,
	double* deriv,
	uint64_t batch_size_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadic_order_1_,
	uint64_t dyadic_order_2_
);