#include "cppch.h"

void add_time(double* data_in, double* data_out, const uint64_t dimension, const uint64_t length) {
	uint64_t dataInSize = dimension * length;

	double* in_ptr = data_in;
	double* out_ptr = data_out;
	double* in_end = data_in + dataInSize;
	auto pointSize = sizeof(double) * dimension;
	double time = 0.;
	double step = 1. / static_cast<double>(length);

	while (in_ptr < in_end) {
		memcpy(out_ptr, in_ptr, pointSize);
		in_ptr += dimension;
		out_ptr += dimension;
		(*out_ptr) = time;
		++out_ptr;
		time += step;
	}
}