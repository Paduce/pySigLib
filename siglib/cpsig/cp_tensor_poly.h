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
#include "cpsig.h"
#include "macros.h"

// Calculate power
// Return 0 on error (integer overflow)
uint64_t power(uint64_t base, uint64_t exp) noexcept;
void log_from_signature_(
	const double* sig,
	double* log_sig,
	uint64_t dimension,
	uint64_t degree
);

void exp_from_log_(
	const double* log_sig,
	double* sig,
	uint64_t dimension,
	uint64_t degree
);

void log_sig_combine_(
	const double* log_sig1,
	const double* log_sig2,
	double* out,
	uint64_t dimension,
	uint64_t degree
);

void batch_log_sig_combine_(
	const double* log_sig1,
	const double* log_sig2,
	double* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int n_jobs
);

FORCE_INLINE void sig_combine_inplace_(
	double* sig1, 
	const double* sig2, 
	uint64_t degree, 
	const uint64_t* level_index
) {

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			double* result_ptr = sig1 + level_index[target_level];
			const double* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];
			for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				const double* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];
				for (const double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*(result_ptr++) += (*left_ptr) * (*right_ptr);
				}
			}
		}

		//left_level = 0
		double* result_ptr = sig1 + level_index[target_level];
		const double* const right_ptr_upper_bound = sig2 + level_index[target_level + 1];
		for (const double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
			*(result_ptr++) += *right_ptr;
		}
	}

}

FORCE_INLINE void sig_uncombine_linear_inplace_(
	double* sig1, 
	const double* sig2, 
	uint64_t degree, 
	const uint64_t* level_index
) {
	//SIG2 MUST BE THE SIGNATURE OF A LINEAR SEGMENT

	for (int64_t target_level = static_cast<int64_t>(degree); target_level > 0; --target_level) {
		for (int64_t left_level = target_level - 1, right_level = 1;
			left_level > 0;
			--left_level, ++right_level) {

			if (right_level % 2) {

				double* result_ptr = sig1 + level_index[target_level];
				const double* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];
				for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
					const double* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];
					for (const double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
						*(result_ptr++) -= (*left_ptr) * (*right_ptr);
					}
				}
			}
			else {

				double* result_ptr = sig1 + level_index[target_level];
				const double* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];
				for (double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
					const double* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];
					for (const double* right_ptr = sig2 + level_index[right_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
						*(result_ptr++) += (*left_ptr) * (*right_ptr);
					}
				}
			}
		}

		//left_level = 0
		if (target_level % 2) {
			double* result_ptr = sig1 + level_index[target_level];
			const double* const right_ptr_upper_bound = sig2 + level_index[target_level + 1];
			for (const double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
				*(result_ptr++) -= *right_ptr;
			}
		}
		else {
			double* result_ptr = sig1 + level_index[target_level];
			const double* const right_ptr_upper_bound = sig2 + level_index[target_level + 1];
			for (const double* right_ptr = sig2 + level_index[target_level]; right_ptr != right_ptr_upper_bound; ++right_ptr) {
				*(result_ptr++) += *right_ptr;
			}
		}
	}

}

FORCE_INLINE void uncombine_sig_deriv(
	const double* sig1,
	const double* sig2,
	double* sig_concat_deriv, 
	double* sig2_deriv,
	uint64_t dimension,
	uint64_t degree, 
	const uint64_t* level_index
) {
	//sig1, sig2 are two signatures, and sig_concat is
	//the signature of the concatenated paths, sig1 * sig2.
	//sig_concat_deriv is dF/d(sig_concat)
	//This function computes dF/d(sig1) and dF/d(sig2) and writes these
	//into sig_concat_deriv and sig2_deriv respectively

	const uint64_t sig_len_ = sig_length(dimension, degree);
	std::memcpy(sig2_deriv, sig_concat_deriv, sig_len_ * sizeof(double));

	for (uint64_t level = degree; level > 0; --level) {
		for (uint64_t left_level = level - 1, right_level = 1; left_level > 0; --left_level, ++right_level) {
			double* result_ptr = sig_concat_deriv + level_index[level];
			double* right_ptr_ = sig2_deriv + level_index[right_level];
			const double* const right_ptr_upper_bound = sig2_deriv + level_index[right_level + 1];
			const double* const left_ptr_upper_bound = sig1 + level_index[left_level + 1];

			for (const double* left_ptr = sig1 + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				for (double* right_ptr = right_ptr_; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*right_ptr += *(result_ptr++) * *left_ptr;
				}
			}
		}
	}


	for (uint64_t left_level = 1; left_level < degree; ++left_level) {
		for (uint64_t level = left_level + 1, right_level = 1; level <= degree; ++level, ++right_level) {
			double* result_ptr = sig_concat_deriv + level_index[level];
			const double* const left_ptr_upper_bound = sig_concat_deriv + level_index[left_level + 1];
			const double* right_ptr_ = sig2 + level_index[right_level];
			const double* const right_ptr_upper_bound = sig2 + level_index[right_level + 1];

			for (double* left_ptr = sig_concat_deriv + level_index[left_level]; left_ptr != left_ptr_upper_bound; ++left_ptr) {
				for (const double* right_ptr = right_ptr_; right_ptr != right_ptr_upper_bound; ++right_ptr) {
					*left_ptr += *(result_ptr++) * (*right_ptr);
				}
			}
		}
	}

}

FORCE_INLINE void linear_sig_deriv_to_increment_deriv(
	const double* sig,
	double* sig_deriv,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	//Given sig is the signature of a line segment [a,b] and sig_deriv
	//is the derivative dF/d(sig), then this function computes dF/d(b-a)
	// and writes it into sig_deriv[1:1+dimension].

	for (uint64_t level = degree; level > 1; --level) {
		const double one_over_level = 1. / level;
		const uint64_t level_size = level_index[level] - level_index[level - 1];
		for (uint64_t j = 0; j < level_size; ++j) {
			const uint64_t offs1 = level_index[level] + dimension * j - 1;
			const uint64_t offs2 = level_index[level - 1] + j;
			for (uint64_t dd = 1; dd <= dimension; ++dd) {
				const double ii = sig_deriv[offs1 + dd] * one_over_level;
				sig_deriv[offs2] += sig[dd] * ii;
				sig_deriv[dd] += sig[offs2] * ii;
			}
		}
	}
}
