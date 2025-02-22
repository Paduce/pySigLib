#pragma once
#include "cppch.h"

#include "multithreading.h"

#include "cpPath.h"
#include "macros.h"
#ifdef AVX
#include "cpVectorFuncs.h"
#endif

template<typename T>
void getSigKernel_(
	T* path1,
	T* path2,
	const uint64_t length1,
	const uint64_t length2,
	const uint64_t dimension,
	double* out,
	const uint64_t dyadicOrder1,
	const uint64_t dyadicOrder2
) {

	T* prevPt1(path1);
	T* nextPt1(path1);
	nextPt1 += dimension;

	T* const lastPt1(path1 + dimension * length1);
	T* const lastPt2(path2 + dimension * length2);

	const double dyadicFrac = 1. / (1ULL << (dyadicOrder1 + dyadicOrder2));
	const double twelth = 1. / 12;

	// Dyadically refined grid dimensions
	const uint64_t gridSize1 = 1ULL << dyadicOrder1;
	const uint64_t gridSize2 = 1ULL << dyadicOrder2;
	const uint64_t dyadicLength1 = ((length1 - 1) << dyadicOrder1) + 1;
	const uint64_t dyadicLength2 = ((length2 - 1) << dyadicOrder2) + 1;

	// Allocate(flattened) PDE grid
	double* pdeGrid = (double*)malloc(dyadicLength1 * dyadicLength2 * sizeof(double));

	// Initialization of K array
	for (uint64_t i = 0; i < dyadicLength1; ++i) {
		pdeGrid[i * dyadicLength2] = 1.0; // Set K[i, 0] = 1.0
	}

	std::fill(pdeGrid, pdeGrid + dyadicLength2, 1.0); // Set K[0, j] = 1.0

	double* diff1 = (double*)malloc(dimension * sizeof(double));
	if (!diff1) {
		throw std::bad_alloc();
		return;
	}
	double* diff2 = (double*)malloc(dimension * sizeof(double));
	if (!diff2) {
		throw std::bad_alloc();
		return;
	}
	double* derivTerm1 = (double*)malloc((length2 - 1) * sizeof(double)); // 1.0 + 0.5 * deriv + deriv2
	if (!derivTerm1) {
		throw std::bad_alloc();
		return;
	}
	double* derivTerm2 = (double*)malloc((length2 - 1) * sizeof(double)); // 1.0 - deriv2
	if (!derivTerm2) {
		throw std::bad_alloc();
		return;
	}

	double* k11 = pdeGrid;
	double* k12 = k11 + 1;
	double* k21 = k11 + dyadicLength2;
	double* k22 = k21 + 1;

	for (uint64_t ii = 0; ii < length1 - 1; prevPt1+=dimension, nextPt1 += dimension, ++ii) {

		for (uint64_t k = 0; k < dimension; ++k)
			diff1[k] = static_cast<double>(nextPt1[k]) - static_cast<double>(prevPt1[k]);

		T* prevPt2(path2);
		T* nextPt2(path2);
		nextPt2 += dimension;
		for (uint64_t m = 0; m < length2 - 1; prevPt2 += dimension, nextPt2 += dimension, ++m) {
			//double deriv = 0;
			for (uint64_t k = 0; k < dimension; ++k) {
				diff2[k] = static_cast<double>((nextPt2[k] - prevPt2[k]));
			}
			/*for (uint64_t k = 0; k < dimension; ++k) {
				deriv += diff1[k] * diff2[k];
			}*/
			double deriv = dot_product(diff1, diff2, dimension);
			deriv *= dyadicFrac;
			double deriv2 = deriv * deriv * twelth;
			derivTerm1[m] = 1.0 + 0.5 * deriv + deriv2;
			derivTerm2[m] = 1.0 - deriv2;
		}

		for (uint64_t i = 0; i < gridSize1; ++i) {
			for (uint64_t jj = 0; jj < length2 - 1; ++jj) {
				double t1 = derivTerm1[jj];
				double t2 = derivTerm2[jj];
				for (uint64_t j = 0; j < gridSize2; ++j) {
					*(k22++) = (*(k21++) + *(k12++)) * t1 - *(k11++) * t2;
				}
			}
			++k11;
			++k12;
			++k21;
			++k22;
		}
	}

	*out = pdeGrid[dyadicLength1 * dyadicLength2 - 1];
	free(derivTerm1);
	free(derivTerm2);
	free(diff1);
	free(diff2);
	free(pdeGrid);
}

template<typename T>
void sigKernel_(
	T* path1,
	T* path2,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }
	getSigKernel_(path1, path2, length1, length2, dimension, out, dyadicOrder1, dyadicOrder2);
}

template<typename T>
void batchSigKernel_(
	T* path1,
	T* path2,
	double* out,
	uint64_t batchSize,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2,
	bool parallel = true
) {
	if (dimension == 0) { throw std::invalid_argument("signature kernel received path of dimension 0"); }

	const uint64_t flatPathLength1 = dimension * length1;
	const uint64_t flatPathLength2 = dimension * length2;
	T* const dataEnd1 = path1 + flatPathLength1 * batchSize;

	auto sigKernelFunc = [&](T* path1Ptr, T* path2Ptr, double* outPtr) {
		getSigKernel_(path1Ptr, path2Ptr, length1, length2, dimension, outPtr, dyadicOrder1, dyadicOrder2);
		};

	if (parallel) {
		multiThreadedBatch2(sigKernelFunc, path1, path2, out, batchSize, flatPathLength1, flatPathLength2, 1);
	}
	else {
		T* path1Ptr = path1;
		T* path2Ptr = path2;
		double* outPtr = out;
		for (;
			path1Ptr < dataEnd1;
			path1Ptr += flatPathLength1, path2Ptr += flatPathLength2, ++outPtr) {

			sigKernelFunc(path1Ptr, path2Ptr, outPtr);
		}
	}
	return;
}