#pragma once
#include "cupch.h"

__global__ void goursatPde(
	double* initialCondition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* gram
);

__device__ void goursatPde32(
	double* initialCondition, //This is the top row of the grid, which will be overwritten to become the bottom row of this grid.
	double* diagonals,
	double* gram,
	uint64_t iteration,
	int numThreads
);

void sigKernelCUDA_(
	double* gram,
	double* out,
	uint64_t batchSize_,
	uint64_t dimension_,
	uint64_t length1_,
	uint64_t length2_,
	uint64_t dyadicOrder1_,
	uint64_t dyadicOrder2_
);