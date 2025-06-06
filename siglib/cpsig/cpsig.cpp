#include "cppch.h"
#include "cpsig.h"
#include <iostream>

#include "cp_path.h"
#include "cp_tensor_poly.h"


double get_path_element(double* data_ptr, int data_length, int data_dimension, int length_index, int dim_index) {
	Path<double> path(data_ptr, static_cast<uint64_t>(data_dimension), static_cast<uint64_t>(data_length));
	return path[static_cast<uint64_t>(length_index)][static_cast<uint64_t>(dim_index)];
}
