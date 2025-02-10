#include "cupch.h"
#include "cusig.h"
#include "cuSigKernel.h"

#define SAFE_CALL(function_call)                 \
    try {                                        \
        function_call;                           \
    }                                            \
    catch (std::bad_alloc&) {					 \
		std::cerr << "Failed to allocate memory";\
        return 1;                                \
    }                                            \
    catch (std::invalid_argument& e) {           \
		std::cerr << e.what();					 \
        return 2;                                \
    }                                            \
	catch (std::out_of_range& e) {			     \
		std::cerr << e.what();					 \
		return 3;                                \
	}  											 \
    catch (...) {                                \
		std::cerr << "Unknown exception";		 \
        return 4;                                \
    }                                            \
    return 0;


extern "C" {

	CUSIG_API int sigKernelFloatCUDA(float* path1, float* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool timeAug, bool leadLag) noexcept {
		SAFE_CALL(sigKernelCUDA_<float>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, timeAug, leadLag));
	}

	CUSIG_API int sigKernelDoubleCUDA(double* path1, double* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool timeAug, bool leadLag) noexcept {
		SAFE_CALL(sigKernelCUDA_<double>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, timeAug, leadLag));
	}

	CUSIG_API int sigKernelInt32CUDA(int32_t* path1, int32_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool timeAug, bool leadLag) noexcept {
		SAFE_CALL(sigKernelCUDA_<int32_t>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, timeAug, leadLag));
	}

	CUSIG_API int sigKernelInt64CUDA(int64_t* path1, int64_t* path2, double* out, uint64_t dimension, uint64_t length1, uint64_t length2, uint64_t dyadicOrder1, uint64_t dyadicOrder2, bool timeAug, bool leadLag) noexcept {
		SAFE_CALL(sigKernelCUDA_<int64_t>(path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2, timeAug, leadLag));
	}

}
