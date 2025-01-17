#include "cppch.h"
#include "cpsig.h"
#include "cpSignature.h"
#include "macros.h"


template<typename T>
PointImpl<T>* Path<T>::pointImplFactory(uint64_t index) const {
	if (!_timeAug && !_leadLag)
		return new PointImpl(this, index);
	else if (_timeAug && !_leadLag)
		return new PointImplTimeAug(this, index);
	else if (!_timeAug && _leadLag)
		return new PointImplLeadLag(this, index);
	else
		return new PointImplTimeAugLeadLag(this, index);
}

extern "C" {

	CPSIG_API void signatureFloat(float* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<float>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	CPSIG_API void signatureDouble(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<double>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	CPSIG_API void signatureInt32(int32_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<int32_t>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	CPSIG_API void signatureInt64(int64_t* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner) {
		signature_<int64_t>(path, out, dimension, length, degree, timeAug, leadLag, horner);
	}

	CPSIG_API void batchSignatureFloat(float* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<float>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

	CPSIG_API void batchSignatureDouble(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<double>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

	CPSIG_API void batchSignatureInt32(int32_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<int32_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

	CPSIG_API void batchSignatureInt64(int64_t* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug, bool leadLag, bool horner, bool parallel) {
		batchSignature_<int64_t>(path, out, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel);
	}

}
