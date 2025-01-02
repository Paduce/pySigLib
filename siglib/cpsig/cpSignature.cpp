#include "cppch.h"
#include "cpTensorPoly.h"
#include "cpSignature.h"

void signature(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	signature_<double>(path, out, dimension, length, degree, timeAug, leadLag, horner);
}

void signatureInt(int* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	signature_<int>(path, out, dimension, length, degree, timeAug, leadLag, horner);
}

void batchSignature(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	batchSignature_<double>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner);
}

void batchSignatureInt(int* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
	batchSignature_<int>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner);
}