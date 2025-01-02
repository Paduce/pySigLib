#pragma once
#include "cppch.h"
#include "cpTensorPoly.h"

uint64_t power(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result *= base;
        }
        base *= base;
        exp /= 2;
    }
    return result;
}

uint64_t polyLength(uint64_t dimension, uint64_t degree) {
    if (dimension == 0) {
        return 1;
    }
    else if (dimension == 1) {
        return degree + 1;
    }
    else {
        return (power(dimension, degree + 1) - 1) / (dimension - 1);
    }
}