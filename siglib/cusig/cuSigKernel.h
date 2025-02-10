#pragma once
#include "cupch.h"

template<typename T>
__global__ void getSigKernelCUDA_(
	double* pdeGrid,
	T* path1,
	T* path2,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2
) {
	uint64_t blockId = blockIdx.x;
	uint64_t threadId = threadIdx.x;

	double dyadicFrac = 1. / (1ULL << (dyadicOrder1 + dyadicOrder2));
	double twelth = 1. / 12;

	// Dyadically refined grid dimensions
	uint64_t gridSize1 = 1ULL << dyadicOrder1;
	uint64_t gridSize2 = 1ULL << dyadicOrder2;
	uint64_t dyadicLength1 = ((length1 - 1) << dyadicOrder1) + 1;
	uint64_t dyadicLength2 = ((length2 - 1) << dyadicOrder2) + 1;

	uint64_t numAntiDiag = dyadicLength1 + dyadicLength2 - 1;

	// Initialization of K array
	for (uint64_t i = 0; i < dyadicLength1; ++i) {
		pdeGrid[i * dyadicLength2] = 1.0; // Set K[i, 0] = 1.0
	}

	for (uint64_t i = 0; i < dyadicLength2; ++i) {
		pdeGrid[i] = 1.0; // Set K[i, 0] = 1.0
	}

	for (uint64_t p = 2; p < numAntiDiag; ++p) { // First two antidiagonals are initialised to 1
		uint64_t startj, endj;
		if (dyadicLength1 > p) startj = 1ULL;
		else startj = p - dyadicLength1 + 1;
		if (dyadicLength2 > p) endj = p;
		else endj = dyadicLength2;

		uint64_t j = startj + threadId;

		if (j < endj) {
			if (j == 4) {
				int iii = 0;
			}

			uint64_t i = p - j;  // Calculate corresponding i (since i + j = p)
			uint64_t ii = ((i - 1) >> dyadicOrder1) + 1;
			uint64_t jj = ((j - 1) >> dyadicOrder2) + 1;

			double deriv = 0;
			for (uint64_t k = 0; k < dimension; ++k) {
				deriv += (path1[ii * dimension + k] - path1[(ii - 1) * dimension + k]) * (path2[jj * dimension + k] - path2[(jj - 1) * dimension + k]);
			}
			deriv *= dyadicFrac;
			double deriv2 = deriv * deriv * twelth;

			pdeGrid[i * dyadicLength2 + j] = (pdeGrid[i * dyadicLength2 + j - 1] + pdeGrid[(i - 1) * dyadicLength2 + j]) * (
				1. + 0.5 * deriv + deriv2) - pdeGrid[(i - 1) * dyadicLength2 + j - 1] * (1. - deriv2);

			if (j == 4) {
				int iii = 0;
			}
		}
		__syncthreads();
	}

	if (threadId == 0)
		*out = pdeGrid[dyadicLength1 * dyadicLength2 - 1];
}

template<typename T>
void sigKernelCUDA_(
	T* path1,
	T* path2,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2,
	bool timeAug = false,
	bool leadLag = false
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t dyadicLength1 = ((length1 - 1) << dyadicOrder1) + 1;
	const uint64_t dyadicLength2 = ((length2 - 1) << dyadicOrder2) + 1;

	// Allocate(flattened) PDE grid
	double* pdeGrid;
	cudaMalloc((void**)&pdeGrid, dyadicLength1 * dyadicLength2 * sizeof(double));

	getSigKernelCUDA_<<<1, dyadicLength2>>>(pdeGrid, path1, path2, out, dimension, length1, length2, dyadicOrder1, dyadicOrder2);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	cudaFree(pdeGrid);
}

//template<typename T>
//void batchSigKernelCUDA_(
//	T* path1,
//	T* path2,
//	double* out,
//	uint64_t batchSize,
//	uint64_t dimension,
//	uint64_t length1,
//	uint64_t length2,
//	uint64_t dyadicOrder1,
//	uint64_t dyadicOrder2,
//	bool timeAug = false,
//	bool leadLag = false,
//	bool parallel = true
//) {
//	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
//
//	const uint64_t flatPathLength1 = dimension * length1;
//	const uint64_t flatPathLength2 = dimension * length2;
//	T* const dataEnd1 = path1 + flatPathLength1 * batchSize;
//
//	auto sigKernelFunc = [&](T* pathPtr, double* outPtr) {
//		Path<T> pathObj1(path1, dimension, length1, timeAug, leadLag);
//		Path<T> pathObj2(path2, dimension, length2, timeAug, leadLag);
//		getSigKernel_(pathObj1, pathObj2, outPtr, dyadicOrder1, dyadicOrder2);
//		};
//
//	if (parallel) {
//		multiThreadedBatch2(sigKernelFunc, path1, path2, out, batchSize, flatPathLength1, flatPathLength2, 1);
//	}
//	else {
//		for (T* path1Ptr = path1, path2Ptr = path2, outPtr = out;
//			path1Ptr < dataEnd1;
//			path1Ptr += flatPathLength1, path2Ptr += flatPathLength2, ++outPtr) {
//
//			sigKernelFunc(path1Ptr, path2Ptr, outPtr);
//		}
//	}
//	return;
//}