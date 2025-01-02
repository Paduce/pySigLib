#pragma once
#include <cstdint>

#ifdef CPSIG_EXPORTS
#define CPSIG_API __declspec(dllexport)
#else
#define CPSIG_API __declspec(dllimport)
#endif

extern "C" CPSIG_API void cpsig_hello_world(const long x);
extern "C" CPSIG_API uint64_t polyLength(uint64_t dimension, uint64_t degree);
extern "C" CPSIG_API double getPathElement(double* dataPtr, int dataLength, int dataDimension, int lengthIndex, int dimIndex);

extern "C" CPSIG_API void signature(double* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);
extern "C" CPSIG_API void signatureInt(int* path, double* out, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);
extern "C" CPSIG_API void batchSignature(double* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);
extern "C" CPSIG_API void batchSignatureInt(int* path, double* out, uint64_t batchSize, uint64_t dimension, uint64_t length, uint64_t degree, bool timeAug = false, bool leadLag = false, bool horner = true);