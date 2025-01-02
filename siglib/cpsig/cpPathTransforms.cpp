#include "cppch.h"

void addTime(double* dataIn, double* dataOut, const uint64_t dimension, const uint64_t length) {
	uint64_t dataInSize = dimension * length;
	uint64_t dataOutSize = dataInSize + length;

	double* inPtr = dataIn;
	double* outPtr = dataOut;
	double* inEnd = dataIn + dataInSize;
	auto pointSize = sizeof(double) * dimension;
	double time = 0.;
	double step = 1. / length;

	while (inPtr < inEnd) {
		memcpy(outPtr, inPtr, pointSize);
		inPtr += dimension;
		outPtr += dimension;
		(*outPtr) = time;
		++outPtr;
		time += step;
	}
}