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
#include "cp_path_transforms.h"

extern "C" {

	CPSIG_API int transform_path_float(float* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(transform_path_<float>(data_in, data_out, dimension, length, time_aug, lead_lag));
	}

	CPSIG_API int transform_path_double(double* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(transform_path_<double>(data_in, data_out, dimension, length, time_aug, lead_lag));
	}

	CPSIG_API int transform_path_int32(int32_t* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(transform_path_<int32_t>(data_in, data_out, dimension, length, time_aug, lead_lag));
	}

	CPSIG_API int transform_path_int64(int64_t* data_in, double* data_out, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag) noexcept {
		SAFE_CALL(transform_path_<int64_t>(data_in, data_out, dimension, length, time_aug, lead_lag));
	}

	CPSIG_API int batch_transform_path_float(float* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<float>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, n_jobs));
	}

	CPSIG_API int batch_transform_path_double(double* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<double>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, n_jobs));
	}

	CPSIG_API int batch_transform_path_int32(int32_t* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<int32_t>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, n_jobs));
	}

	CPSIG_API int batch_transform_path_int64(int64_t* data_in, double* data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, bool time_aug, bool lead_lag, int n_jobs) noexcept {
		SAFE_CALL(batch_transform_path_<int64_t>(data_in, data_out, batch_size, dimension, length, time_aug, lead_lag, n_jobs));
	}
}