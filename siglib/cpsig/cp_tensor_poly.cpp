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

#include "cppch.h"
#include "cpsig.h"
#include "cp_tensor_poly.h"
#include "multithreading.h"
#include "macros.h"

uint64_t power(uint64_t base, uint64_t exp) noexcept {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            const auto _res = result * base;
            if (_res < result)
                return 0; // overflow
            result = _res;
        }
        const auto _base = base * base;
        if (_base < base)
            return 0; // overflow
        base = _base;
        exp /= 2;
    }
    return result;
}

extern "C" CPSIG_API uint64_t sig_length(uint64_t dimension, uint64_t degree) noexcept {
	if (dimension == 0) {
		return 1;
	}
	else if (dimension == 1) {
        return degree + 1;
    }
    else {
        const auto pwr = power(dimension, degree + 1);
        if (pwr)
            return (pwr - 1) / (dimension - 1);
        else
            return 0; // overflow
	}
}

extern "C" CPSIG_API uint64_t log_sig_length(uint64_t dimension, uint64_t degree) noexcept {
	const auto sig_len = sig_length(dimension, degree);
	if (sig_len == 0) {
		return 0;
	}
	return sig_len - 1;
}

static void fill_level_index(uint64_t dimension, uint64_t degree, uint64_t* level_index) {
	level_index[0] = 0;
	for (uint64_t i = 1; i <= degree + 1; ++i) {
		level_index[i] = level_index[i - 1] * dimension + 1;
	}
}

static void tensor_product_(
	const double* sig1,
	const double* sig2,
	double* out,
	uint64_t dimension,
	uint64_t degree,
	const uint64_t* level_index
) {
	const uint64_t total_length = level_index[degree + 1];
	std::fill(out, out + total_length, 0.);

	// level = 0
	out[0] = sig1[0] * sig2[0];

	for (uint64_t target_level = 1; target_level <= degree; ++target_level) {
		const uint64_t level_size = level_index[target_level + 1] - level_index[target_level];
		double* level_out = out + level_index[target_level];
		std::fill(level_out, level_out + level_size, 0.);

		for (uint64_t left_level = 0; left_level <= target_level; ++left_level) {
			const uint64_t right_level = target_level - left_level;

			const uint64_t left_level_size = level_index[left_level + 1] - level_index[left_level];
			const uint64_t right_level_size = level_index[right_level + 1] - level_index[right_level];

			const double* left_ptr = sig1 + level_index[left_level];
			const double* right_ptr = sig2 + level_index[right_level];

			for (uint64_t left_idx = 0; left_idx < left_level_size; ++left_idx) {
				double* result_ptr = level_out + left_idx * right_level_size;
				const double left_val = left_ptr[left_idx];
				for (uint64_t right_idx = 0; right_idx < right_level_size; ++right_idx) {
					result_ptr[right_idx] += left_val * right_ptr[right_idx];
				}
			}
		}
	}
}

void log_from_signature_(
	const double* sig,
	double* log_sig,
	uint64_t dimension,
	uint64_t degree
) {
	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	fill_level_index(dimension, degree, level_index);

	const uint64_t sig_len = level_index[degree + 1];

	auto y_uptr = std::make_unique<double[]>(sig_len);
	double* y = y_uptr.get();
	y[0] = sig[0] - 1.;
	for (uint64_t i = 1; i < sig_len; ++i) {
		y[i] = sig[i];
	}

	auto pow_curr_uptr = std::make_unique<double[]>(sig_len);
	double* pow_curr = pow_curr_uptr.get();
	std::memcpy(pow_curr, y, sizeof(double) * sig_len);

	auto pow_next_uptr = std::make_unique<double[]>(sig_len);
	double* pow_next = pow_next_uptr.get();

	auto log_full_uptr = std::make_unique<double[]>(sig_len);
	double* log_full = log_full_uptr.get();
	std::fill(log_full, log_full + sig_len, 0.);

	for (uint64_t k = 1; k <= degree; ++k) {
		const double sign = (k % 2) ? 1. : -1.;
		const double coeff = sign / static_cast<double>(k);

		for (uint64_t level = 1; level <= degree; ++level) {
			const uint64_t start = level_index[level];
			const uint64_t end = level_index[level + 1];
			for (uint64_t idx = start; idx < end; ++idx) {
				log_full[idx] += coeff * pow_curr[idx];
			}
		}

		if (k != degree) {
			tensor_product_(pow_curr, y, pow_next, dimension, degree, level_index);
			std::swap(pow_curr, pow_next);
		}
	}

	double* out_ptr = log_sig;
	for (uint64_t level = 1; level <= degree; ++level) {
		const uint64_t start = level_index[level];
		const uint64_t end = level_index[level + 1];
		const uint64_t level_size = end - start;
		std::memcpy(out_ptr, log_full + start, sizeof(double) * level_size);
		out_ptr += level_size;
	}
}

void exp_from_log_(
	const double* log_sig,
	double* sig,
	uint64_t dimension,
	uint64_t degree
) {
	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();
	fill_level_index(dimension, degree, level_index);

	const uint64_t sig_len = level_index[degree + 1];
	std::fill(sig, sig + sig_len, 0.);
	sig[0] = 1.;

	auto x_uptr = std::make_unique<double[]>(sig_len);
	double* x = x_uptr.get();
	std::fill(x, x + sig_len, 0.);

	const double* log_ptr = log_sig;
	for (uint64_t level = 1; level <= degree; ++level) {
		const uint64_t start = level_index[level];
		const uint64_t end = level_index[level + 1];
		const uint64_t level_size = end - start;
		std::memcpy(x + start, log_ptr, sizeof(double) * level_size);
		log_ptr += level_size;
	}

	auto pow_curr_uptr = std::make_unique<double[]>(sig_len);
	double* pow_curr = pow_curr_uptr.get();
	std::memcpy(pow_curr, x, sizeof(double) * sig_len);

	auto pow_next_uptr = std::make_unique<double[]>(sig_len);
	double* pow_next = pow_next_uptr.get();

	double factorial = 1.;
	for (uint64_t k = 1; k <= degree; ++k) {
		factorial *= static_cast<double>(k);
		const double coeff = 1. / factorial;

		for (uint64_t level = 1; level <= degree; ++level) {
			const uint64_t start = level_index[level];
			const uint64_t end = level_index[level + 1];
			for (uint64_t idx = start; idx < end; ++idx) {
				sig[idx] += coeff * pow_curr[idx];
			}
		}

		if (k != degree) {
			tensor_product_(pow_curr, x, pow_next, dimension, degree, level_index);
			std::swap(pow_curr, pow_next);
		}
	}
}

void log_sig_combine_(
	const double* log_sig1,
	const double* log_sig2,
	double* out,
	uint64_t dimension,
	uint64_t degree
) {
	if (dimension == 0) { throw std::invalid_argument("log_sig_combine received dimension 0"); }

	const uint64_t sig_len = ::sig_length(dimension, degree);
	auto sig1_uptr = std::make_unique<double[]>(sig_len);
	auto sig2_uptr = std::make_unique<double[]>(sig_len);
	auto sig_concat_uptr = std::make_unique<double[]>(sig_len);

	exp_from_log_(log_sig1, sig1_uptr.get(), dimension, degree);
	exp_from_log_(log_sig2, sig2_uptr.get(), dimension, degree);

	sig_combine_(sig1_uptr.get(), sig2_uptr.get(), sig_concat_uptr.get(), dimension, degree);

	log_from_signature_(sig_concat_uptr.get(), out, dimension, degree);
}

void batch_log_sig_combine_(
	const double* log_sig1,
	const double* log_sig2,
	double* out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t degree,
	int n_jobs
) {
	if (dimension == 0) { throw std::invalid_argument("log_sig_combine received dimension 0"); }

	const uint64_t log_sig_len = ::log_sig_length(dimension, degree);
	const double* const log_sig1_end = log_sig1 + log_sig_len * batch_size;

	std::function<void(const double*, const double*, double*)> log_sig_combine_func;
	log_sig_combine_func = [&](const double* log_sig1_ptr, const double* log_sig2_ptr, double* out_ptr) {
		log_sig_combine_(log_sig1_ptr, log_sig2_ptr, out_ptr, dimension, degree);
	};

	if (n_jobs != 1) {
		multi_threaded_batch_2(log_sig_combine_func, log_sig1, log_sig2, out, batch_size, log_sig_len, log_sig_len, log_sig_len, n_jobs);
	}
	else {
		const double* log_sig1_ptr = log_sig1;
		const double* log_sig2_ptr = log_sig2;
		double* out_ptr = out;
		for (; log_sig1_ptr < log_sig1_end;
			log_sig1_ptr += log_sig_len,
			log_sig2_ptr += log_sig_len,
			out_ptr += log_sig_len) {

			log_sig_combine_func(log_sig1_ptr, log_sig2_ptr, out_ptr);
		}
	}
}


void sig_combine_(
	const double* sig1,
	const double* sig2,
	double* out, 
	uint64_t dimension, 
	uint64_t degree
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();

	level_index[0] = 0;
	for (uint64_t i = 1; i <= degree + 1; i++)
		level_index[i] = level_index[i - 1] * dimension + 1;

    std::memcpy(out, sig1, sizeof(double) * level_index[degree + 1]);

	sig_combine_inplace_(out, sig2, degree, level_index);
}

void batch_sig_combine_(
	const double* sig1,
	const double* sig2,
	double* out, 
	uint64_t batch_size,
	uint64_t dimension, 
	uint64_t degree, 
	int n_jobs = 1
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine received dimension 0"); }

	const uint64_t siglength = ::sig_length(dimension, degree);
	const double* const sig1_end = sig1 + siglength * batch_size;

	std::function<void(const double*, const double*, double*)> sig_combine_func;

	sig_combine_func = [&](const double* sig1_ptr, const double* sig2_ptr, double* out_ptr) {
		sig_combine_(sig1_ptr, sig2_ptr, out_ptr, dimension, degree);
		};

	if (n_jobs != 1) {
		multi_threaded_batch_2(sig_combine_func, sig1, sig2, out, batch_size, siglength, siglength, siglength, n_jobs);
	}
	else {
		const double* sig1_ptr = sig1;
		const double* sig2_ptr = sig2;
		double* out_ptr = out;
		for (;
			sig1_ptr < sig1_end;
			sig1_ptr += siglength,
			sig2_ptr += siglength,
			out_ptr += siglength) {

			sig_combine_func(sig1_ptr, sig2_ptr, out_ptr);
		}
	}
	return;
}

void sig_combine_backprop_(
	const double* sig_combined_deriv,
	double* sig1_deriv, 
	double* sig2_deriv, 
	const double* sig1,
	const double* sig2,
	uint64_t dimension,
	uint64_t degree
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine_backprop received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();

	level_index[0] = 0;
	for (uint64_t i = 1; i <= degree + 1; i++)
		level_index[i] = level_index[i - 1] * dimension + 1;

	std::memcpy(sig1_deriv, sig_combined_deriv, sizeof(double) * level_index[degree + 1]);

	uncombine_sig_deriv(sig1, sig2, sig1_deriv, sig2_deriv, dimension, degree, level_index);
	return;
}

void batch_sig_combine_backprop_(
	const double* sig_combined_deriv,
	double* sig1_deriv, 
	double* sig2_deriv, 
	const double* sig1,
	const double* sig2,
	uint64_t batch_size,
	uint64_t dimension, 
	uint64_t degree,
	int n_jobs = 1
)
{
	if (dimension == 0) { throw std::invalid_argument("sig_combine_backprop received dimension 0"); }

	auto level_index_uptr = std::make_unique<uint64_t[]>(degree + 2);
	uint64_t* level_index = level_index_uptr.get();

	level_index[0] = 0;
	for (uint64_t i = 1; i <= degree + 1; i++)
		level_index[i] = level_index[i - 1] * dimension + 1;

	const uint64_t siglength = level_index[degree + 1];

	std::memcpy(sig1_deriv, sig_combined_deriv, sizeof(double) * siglength * batch_size);

	std::function<void(const double*, double*, double*, const double*, const double*)> sig_combine_backprop_func;

	sig_combine_backprop_func = [&](const double* sig_combined_deriv_ptr, double* sig1_deriv_ptr, double* sig2_deriv_ptr, const double* sig1_ptr, const double* sig2_ptr) {
		sig_combine_backprop_(sig_combined_deriv_ptr, sig1_deriv_ptr, sig2_deriv_ptr, sig1_ptr, sig2_ptr, dimension, degree);
		};

	if (n_jobs != 1) {
		multi_threaded_batch_4(sig_combine_backprop_func, sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, siglength, siglength, siglength, siglength, siglength, n_jobs);
	}
	else {
		const double* sig_combined_derivs_ptr = sig_combined_deriv;
		double* sig1_deriv_ptr = sig1_deriv;
		double* sig2_deriv_ptr = sig2_deriv;
		const double* sig1_ptr = sig1;
		const double* sig2_ptr = sig2;
		const double* sig1_end = sig1 + batch_size * siglength;
		for (;
			sig1_ptr < sig1_end;
			sig_combined_derivs_ptr += siglength,
			sig1_deriv_ptr += siglength,
			sig2_deriv_ptr += siglength,
			sig1_ptr += siglength,
			sig2_ptr += siglength
			) {

			sig_combine_backprop_func(sig_combined_derivs_ptr, sig1_deriv_ptr, sig2_deriv_ptr, sig1_ptr, sig2_ptr);
		}
	}
	return;
}

extern "C" {

	CPSIG_API int sig_combine(const double* sig1, const double* sig2, double* out, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(sig_combine_(sig1, sig2, out, dimension, degree));
	}

	CPSIG_API int log_sig_combine(const double* log_sig1, const double* log_sig2, double* out, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(log_sig_combine_(log_sig1, log_sig2, out, dimension, degree));
	}

	CPSIG_API int batch_sig_combine(const double* sig1, const double* sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_combine_(sig1, sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int batch_log_sig_combine(const double* log_sig1, const double* log_sig2, double* out, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_log_sig_combine_(log_sig1, log_sig2, out, batch_size, dimension, degree, n_jobs));
	}

	CPSIG_API int sig_combine_backprop(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t dimension, uint64_t degree) noexcept {
		SAFE_CALL(sig_combine_backprop_(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, dimension, degree));
	}

	CPSIG_API int batch_sig_combine_backprop(const double* sig_combined_deriv, double* sig1_deriv, double* sig2_deriv, const double* sig1, const double* sig2, uint64_t batch_size, uint64_t dimension, uint64_t degree, int n_jobs) noexcept {
		SAFE_CALL(batch_sig_combine_backprop_(sig_combined_deriv, sig1_deriv, sig2_deriv, sig1, sig2, batch_size, dimension, degree, n_jobs));
	}
}
