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
#include "macros.h"
#include "cp_path.h"
#include "multithreading.h"

template<typename T>
void transform_path_(
	const T* data_in,
	double* data_out,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	double end_time
) {
	// Note that here 'dimension' and 'length' refer to the underlying path, not the transformed one.
	// Also note that if both time_aug and lead_lag are false, we still copy memory to data_out

	const Path<double> dummy_path(nullptr, dimension, length, time_aug, lead_lag, end_time);
	const uint64_t transformed_dimension = dummy_path.dimension();
	const uint64_t transformed_length = dummy_path.length();

	if (!(time_aug || lead_lag)) {
		const uint64_t path_size = dimension * length;
		double* out_ptr = data_out;
		const T* in_ptr = data_in;
		for (uint64_t i = 0; i < path_size; ++i)
			*(out_ptr++) = static_cast<double>(*(in_ptr++));
		return;
	}

	if (lead_lag) {
		const uint64_t twice_dimension = 2 * dimension;
		uint64_t dim_update = 2 * transformed_dimension - dimension;

		double* out_ptr = data_out;
		double* out_ptr_2 = data_out + dimension;
		const T* in_ptr = data_in;

		for (uint64_t i = 0; i < length; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) = static_cast<double>(*in_ptr);
				*(out_ptr_2++) = static_cast<double>(*(in_ptr++));
			}
			out_ptr += dim_update;
			out_ptr_2 += dim_update;
		}

		dim_update = 2 * transformed_dimension - twice_dimension;
		out_ptr = data_out + transformed_dimension;
		in_ptr = data_in;

		for (uint64_t i = 0; i < length - 1; ++i) {
			for (uint64_t j = 0; j < twice_dimension; ++j) {
				*(out_ptr++) = static_cast<double>(*(in_ptr++));
			}
			out_ptr += dim_update;
			in_ptr -= dimension;
		}
	}
	else {
		const uint64_t dim_update = transformed_dimension - dimension;
		double* out_ptr = data_out;
		const T* in_ptr = data_in;

		for (uint64_t i = 0; i < length; ++i) {
			for (uint64_t j = 0; j < dimension; ++j) {
				*(out_ptr++) = static_cast<double>(*(in_ptr++));
			}
			out_ptr += dim_update;
		}
	}

	if (time_aug) {
		double* const out_ptr = data_out + transformed_dimension - 1;
		const double scale = end_time / (transformed_length - 1);

		for (uint64_t i = 0; i < transformed_length; ++i) {
			out_ptr[i * transformed_dimension] = i * scale;
		}
	}

}

template<typename T>
void batch_transform_path_(
	const T* data_in,
	double* data_out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	double end_time,
	int n_jobs
)
{
	//Deal with trivial cases
	if (dimension == 0) { throw std::invalid_argument("transform_path received path of dimension 0"); }

	const Path<T> dummy_path_obj(nullptr, dimension, length, time_aug, lead_lag, end_time); //Work with path_obj to capture time_aug, lead_lag transformations

	const uint64_t result_length = dummy_path_obj.length() * dummy_path_obj.dimension();

	const uint64_t flat_path_length = dimension * length;
	const T* const data_end = data_in + flat_path_length * batch_size;

	std::function<void(const T* const, double* const)> transform_func;

	transform_func = [&](const T* const path_ptr, double* const out_ptr) {
		transform_path_<T>(path_ptr, out_ptr, dimension, length, time_aug, lead_lag, end_time);
		};

	const T* path_ptr;
	double* out_ptr;

	if (n_jobs != 1) {
		multi_threaded_batch(transform_func, data_in, data_out, batch_size, flat_path_length, result_length, n_jobs);
	}
	else {
		for (path_ptr = data_in, out_ptr = data_out;
			path_ptr < data_end;
			path_ptr += flat_path_length, out_ptr += result_length) {

			transform_func(path_ptr, out_ptr);
		}
	}
	return;
}

void transform_path_backprop_(
	const double* data_in,
	double* data_out,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	double end_time
);


void batch_transform_path_backprop_(
	const double* data_in,
	double* data_out,
	uint64_t batch_size,
	uint64_t dimension,
	uint64_t length,
	bool time_aug,
	bool lead_lag,
	double end_time,
	int n_jobs
);
