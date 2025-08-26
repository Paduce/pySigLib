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

#ifdef CUSIG_EXPORTS
#ifdef _WIN32
#define CUSIG_API __declspec(dllexport)
#else
#define CUSIG_API
#endif
#else
#define CUSIG_API __declspec(dllimport)
#endif

extern "C" {
	CUSIG_API int transform_path_cuda_float(const float* const data_in, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int transform_path_cuda_double(const double* const data_in, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int transform_path_cuda_int32(const int32_t* const data_in, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int transform_path_cuda_int64(const int64_t* const data_in, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int batch_transform_path_cuda_float(const float* const data_in, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int batch_transform_path_cuda_double(const double* const data_in, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int batch_transform_path_cuda_int32(const int32_t* const data_in, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int batch_transform_path_cuda_int64(const int64_t* const data_in, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int transform_path_backprop_cuda(const double* const derivs, double* const data_out, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;
	CUSIG_API int batch_transform_path_backprop_cuda(const double* const derivs, double* const data_out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length, const bool time_aug, const bool lead_lag, const double end_time) noexcept;

	CUSIG_API int sig_kernel_cuda(const double* const gram, double* const out, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept;
	CUSIG_API int batch_sig_kernel_cuda(const double* const gram, double* const out, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2, const bool return_grid) noexcept;
	CUSIG_API int sig_kernel_backprop_cuda(const double* const gram, double* const out, const double deriv, const double* const k_grid, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2) noexcept;
	CUSIG_API int batch_sig_kernel_backprop_cuda(const double* const gram, double* const out, const double* const deriv, const double* const k_grid, const uint64_t batch_size, const uint64_t dimension, const uint64_t length1, const uint64_t length2, const uint64_t dyadic_order_1, const uint64_t dyadic_order_2) noexcept;
}
