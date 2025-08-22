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
#include "cppch.h"

#include "multithreading.h"

#include "cp_path.h"
#include "macros.h"
#ifdef VEC
#include "cp_vector_funcs.h"
#endif

void get_a_b(double& a, double& b, const double* const gram, uint64_t ii, uint64_t jj, const uint64_t length2, const double dyadic_frac);
void get_a(double& a, const double* const gram, uint64_t ii, uint64_t jj, const uint64_t length2, const double dyadic_frac);
void get_b(double& b, const double* const gram, uint64_t ii, uint64_t jj, const uint64_t length2, const double dyadic_frac);
void get_a_b_deriv(double& a_deriv, double& b_deriv, const double* const gram, uint64_t ii, uint64_t jj, const uint64_t length2, const double dyadic_frac);

void get_sig_kernel_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2,
	bool return_grid
);

template<bool order>//order is True if dyadic_length_2 <= dyadic_length_1
void get_sig_kernel_diag_internal_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2,
	const uint64_t dyadic_length_1,
	const uint64_t dyadic_length_2
) {
	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const double twelth = 1. / 12;
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<double[]>(diag_len * 3);
	double* diagonals = diagonals_uptr.get();

	double* prev_prev_diag = diagonals;
	double* prev_diag = diagonals + diag_len;
	double* next_diag = diagonals + 2 * diag_len;

	// Initialization
	std::fill(diagonals, diagonals + 3 * diag_len, 1.);

	for (uint64_t p = 2; p < num_anti_diag; ++p) { // First two antidiagonals are initialised to 1

		if (order) {
			uint64_t startj, endj;
			if (dyadic_length_1 > p) startj = 1ULL;
			else startj = p - dyadic_length_1 + 1;
			if (dyadic_length_2 > p) endj = p;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_1);
				uint64_t jj = ((j - 1) >> dyadic_order_2);

				double deriv = gram[ii * (length2 - 1) + jj];
				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					1. + 0.5 * deriv + deriv2) - *(prev_prev_diag + j - 1) * (1. - deriv2);

			}
		}
		else {
			uint64_t startj, endj;
			if (dyadic_length_2 > p) startj = 1ULL;
			else startj = p - dyadic_length_2 + 1;
			if (dyadic_length_1 > p) endj = p;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				uint64_t ii = ((i - 1) >> dyadic_order_2);
				uint64_t jj = ((j - 1) >> dyadic_order_1);

				double deriv = gram[jj * (length2 - 1) + ii];

				deriv *= dyadic_frac;
				double deriv2 = deriv * deriv * twelth;

				*(next_diag + j) = (*(prev_diag + j) + *(prev_diag + j - 1)) * (
					1. + 0.5 * deriv + deriv2) - *(prev_prev_diag + j - 1) * (1. - deriv2);

			}
		}

		// Rotate the diagonals (swap pointers, no data copying)
		double* temp = prev_prev_diag;
		prev_prev_diag = prev_diag;
		prev_diag = next_diag;
		next_diag = temp;
	}

	*out = prev_diag[diag_len - 1];
}

template<bool order>//order is True if dyadic_length_2 <= dyadic_length_1
void get_sig_kernel_backprop_diag_internal_(
	const double* const gram,
	double* const out,
	const double deriv,
	double* k_grid,
	const uint64_t dimension,
	const uint64_t length1,
	const uint64_t length2,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2,
	const uint64_t dyadic_length_1,
	const uint64_t dyadic_length_2
) {
	// General structure of the grids:
	// 
	// dF / dk = 0 for the first row and column of k_grid, so disregard these.
	// Flip the remaining grid, so that the last element is now in the top left.
	// Now, add a row and column of zeros as initial conditions to the grid, such that it now
	// has the same dimensions as k_grid.
	// The resulting grid is what is traversed by 'diagonals' below.
	// 
	// The grids for A, B, dA and dB are flipped and padded similarly, such that
	// the value at index [1,1] is the value at [-1,-1] in the original grids.
	// We will only need one diagonal for each of these, containing the values
	// Needed to update the leading diagonal of dF / dk. Note that for A, these
	// values are lagged, i.e. we need values A(i-1,j) and A(i,j-1) to update
	// dF / dk(i,j). Similarly dA and dB are lagged by two, as we need
	// dA(i-1, j-1) and dB(i-1, j-1).

	const double dyadic_frac = 1. / (1ULL << (dyadic_order_1 + dyadic_order_2));
	const double twelth = 1. / 12;
	const uint64_t num_anti_diag = dyadic_length_1 + dyadic_length_2 - 1;
	const uint64_t grid_length = dyadic_length_1 * dyadic_length_2;
	const uint64_t gram_length = (length1 - 1) * (length2 - 1);

	// Allocate three diagonals
	const uint64_t diag_len = std::min(dyadic_length_1, dyadic_length_2);
	auto diagonals_uptr = std::make_unique<double[]>(diag_len * 3);
	double* diagonals = diagonals_uptr.get();

	// Allocate diagonals to store A, B, A_deriv, B_deriv
	auto a_uptr = std::make_unique<double[]>(diag_len);
	double* a = a_uptr.get();

	auto b_uptr = std::make_unique<double[]>(diag_len);
	double* b = b_uptr.get();

	auto da_uptr = std::make_unique<double[]>(diag_len);
	double* da = da_uptr.get();

	auto db_uptr = std::make_unique<double[]>(diag_len);
	double* db = db_uptr.get();

	// Indicies for diagonals
	uint64_t prev_prev_diag_idx = 0;
	uint64_t prev_diag_idx = diag_len;
	uint64_t next_diag_idx = 2 * diag_len;

	// Initialization
	std::fill(out, out + (length1 - 1) * (length2 - 1), 0.);
	std::fill(diagonals, diagonals + 3 * diag_len, 0.);
	std::fill(a, a + diag_len, 0.);
	std::fill(b, b + diag_len, 0.);
	std::fill(da, da + diag_len, 0.);
	std::fill(db, db + diag_len, 0.);
	
	diagonals[prev_diag_idx + 1] = deriv;
	get_a_b_deriv(da[1], db[1], gram, length1 - 2, length2 - 2, length2, dyadic_frac);

	//Update dF / dx for first value
	out[gram_length - 1] += deriv * (
		(k_grid[grid_length - 2] + k_grid[grid_length - dyadic_length_2 - 1]) * da[1] - k_grid[grid_length - dyadic_length_2 - 2] * db[1]
		);

	for (uint64_t p = 3; p < num_anti_diag; ++p) { // First three antidiagonals are initialised

		if (order) {

			//Update b
			uint64_t startj, endj;
			int64_t p_ = p - 2;
			if (dyadic_length_1 > p_) startj = 1ULL;
			else startj = p_ - dyadic_length_1 + 1;
			if (dyadic_length_2 > p_) endj = p_;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p_ - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_1 - i - 1;
				const uint64_t j_rev = dyadic_length_2 - j - 1;
				const uint64_t ii = (i_rev >> dyadic_order_1);
				const uint64_t jj = (j_rev >> dyadic_order_2);

				get_b(b[j], gram, ii, jj, length2, dyadic_frac);
			}

			//Update a
			p_ = p - 1;
			if (dyadic_length_1 > p_) startj = 1ULL;
			else startj = p_ - dyadic_length_1 + 1;
			if (dyadic_length_2 > p_) endj = p_;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p_ - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_1 - i - 1;
				const uint64_t j_rev = dyadic_length_2 - j - 1;
				const uint64_t ii = (i_rev >> dyadic_order_1);
				const uint64_t jj = (j_rev >> dyadic_order_2);

				get_a(a[j], gram, ii, jj, length2, dyadic_frac);
			}

			//Update da, db
			p_ = p;
			if (dyadic_length_1 > p_) startj = 1ULL;
			else startj = p_ - dyadic_length_1 + 1;
			if (dyadic_length_2 > p_) endj = p_;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p_ - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_1 - i - 1;
				const uint64_t j_rev = dyadic_length_2 - j - 1;
				const uint64_t ii = (i_rev >> dyadic_order_1);
				const uint64_t jj = (j_rev >> dyadic_order_2);

				get_a_b_deriv(da[j], db[j], gram, ii, jj, length2, dyadic_frac);
			}

			if (dyadic_length_1 > p) startj = 1ULL;
			else startj = p - dyadic_length_1 + 1;
			if (dyadic_length_2 > p) endj = p;
			else endj = dyadic_length_2;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_1 - i;
				const uint64_t j_rev = dyadic_length_2 - j;
				const uint64_t ii = ((i_rev-1) >> dyadic_order_1);
				const uint64_t jj = ((j_rev-1) >> dyadic_order_2);

				// Update dF / dk
				diagonals[next_diag_idx + j] = diagonals[prev_diag_idx + j - 1] * a[j-1]
					+ diagonals[prev_diag_idx + j] * a[j]
					- diagonals[prev_prev_diag_idx + j - 1] * b[j-1]; //TODO: check indices for a and b here

				// Update dF / dx
				const uint64_t idx = i_rev * dyadic_length_2 + j_rev;
				out[ii * (length2 - 1) + jj] += diagonals[next_diag_idx + j] * (
					(k_grid[idx - 1] + k_grid[idx - dyadic_length_2]) * da[j] - k_grid[idx - dyadic_length_2 - 1] * db[j]
					);
			}
		}
		else {
			//Update b
			uint64_t startj, endj;
			int64_t p_ = p - 2;
			if (dyadic_length_2 > p_) startj = 1ULL;
			else startj = p_ - dyadic_length_2 + 1;
			if (dyadic_length_1 > p_) endj = p_;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p_ - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_2 - i - 1;
				const uint64_t j_rev = dyadic_length_1 - j - 1;
				const uint64_t ii = (i_rev >> dyadic_order_2);
				const uint64_t jj = (j_rev >> dyadic_order_1);

				get_b(b[j], gram, jj, ii, length2, dyadic_frac);
			}

			//Update a
			p_ = p - 1;
			if (dyadic_length_2 > p_) startj = 1ULL;
			else startj = p_ - dyadic_length_2 + 1;
			if (dyadic_length_1 > p_) endj = p_;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p_ - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_2 - i - 1;
				const uint64_t j_rev = dyadic_length_1 - j - 1;
				const uint64_t ii = (i_rev >> dyadic_order_2);
				const uint64_t jj = (j_rev >> dyadic_order_1);

				get_a(a[j], gram, jj, ii, length2, dyadic_frac);
			}

			//Update da, db
			p_ = p;
			if (dyadic_length_2 > p_) startj = 1ULL;
			else startj = p_ - dyadic_length_2 + 1;
			if (dyadic_length_1 > p_) endj = p_;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p_ - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_2 - i - 1;
				const uint64_t j_rev = dyadic_length_1 - j - 1;
				const uint64_t ii = (i_rev >> dyadic_order_2);
				const uint64_t jj = (j_rev >> dyadic_order_1);

				get_a_b_deriv(da[j], db[j], gram, jj, ii, length2, dyadic_frac);
			}

			if (dyadic_length_2 > p) startj = 1ULL;
			else startj = p - dyadic_length_2 + 1;
			if (dyadic_length_1 > p) endj = p;
			else endj = dyadic_length_1;

			for (uint64_t j = startj; j < endj; ++j) {
				const uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
				const uint64_t i_rev = dyadic_length_2 - i;
				const uint64_t j_rev = dyadic_length_1 - j;
				const uint64_t ii = ((i_rev - 1) >> dyadic_order_2);
				const uint64_t jj = ((j_rev - 1) >> dyadic_order_1);

				// Update dF / dk
				diagonals[next_diag_idx + j] = diagonals[prev_diag_idx + j - 1] * a[j - 1]
					+ diagonals[prev_diag_idx + j] * a[j]
					- diagonals[prev_prev_diag_idx + j - 1] * b[j - 1]; //TODO: check indices for a and b here

				// Update dF / dx
				const uint64_t idx = j_rev * dyadic_length_2 + i_rev;
				out[jj * (length2 - 1) + ii] += diagonals[next_diag_idx + j] * (
					(k_grid[idx - 1] + k_grid[idx - dyadic_length_2]) * da[j] - k_grid[idx - dyadic_length_2 - 1] * db[j]
					);
			}
		}

		// Rotate the diagonals (swap pointers, no data copying)
		uint64_t temp_idx = prev_prev_diag_idx;
		prev_prev_diag_idx = prev_diag_idx;
		prev_diag_idx = next_diag_idx;
		next_diag_idx = temp_idx;
	}
}

void get_sig_kernel_diag_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	double* out,
	const uint64_t dyadic_order_1,
	const uint64_t dyadic_order_2
);

void sig_kernel_(
	double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	bool return_grid = false
);

void batch_sig_kernel_(
	double* gram,
	double* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	int n_jobs = 1,
	bool return_grid = false
);

void sig_kernel_backprop_(
	double* gram,
	double* out,
	double deriv,
	double* k_grid,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);

void batch_sig_kernel_backprop_(
	double* gram,
	double* out,
	double* derivs,
	double* k_grid,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2,
	int n_jobs
);

void get_sig_kernel_backprop_diag_(
	double* gram,
	double* out,
	double deriv,
	double* k_grid,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);

void get_sig_kernel_backprop_(
	double* gram,
	double* out,
	double deriv,
	double* k_grid,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadic_order_1,
	uint64_t dyadic_order_2
);