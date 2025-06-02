#pragma once

#ifdef CUSIG_EXPORTS
#define CUSIG_API __declspec(dllexport)
#else
#define CUSIG_API __declspec(dllimport)
#endif

extern "C" CUSIG_API void cusig_hello_world(const long x);

extern "C" {
	CUSIG_API int sigKernelCUDA(double* gram, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
	CUSIG_API int batchSigKernelCUDA(double* gram, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2) noexcept;
}
