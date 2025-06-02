#pragma once
#include "cppch.h"

#include "multithreading.h"

#include "cpPath.h"
#include "macros.h"
#ifdef AVX
#include "cpVectorFuncs.h"
#endif

void getSigKernel_(
	double* gram,
	const uint64_t length1,
	const uint64_t length2,
	const uint64_t dimension,
	double* out,
	const uint64_t dyadicOrder1,
	const uint64_t dyadicOrder2
);

void sigKernel_(
	double* gram,
	double* out,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2
);

void batchSigKernel_(
	double* gram,
	double* out,
	uint64_t batchSize,
	uint64_t dimension,
	uint64_t length1,
	uint64_t length2,
	uint64_t dyadicOrder1,
	uint64_t dyadicOrder2,
	bool parallel = true
);