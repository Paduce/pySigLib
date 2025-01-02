#include "cppch.h"
#include "cpsig.h"
#include <iostream>

#include "cpPath.h"
#include "cpTensorPoly.h"

void cpsig_hello_world(const long x)
{
	std::cout << "cpsig Hello World " + std::to_string(x) << std::endl;
}

double getPathElement(double* dataPtr, int dataLength, int dataDimension, int lengthIndex, int dimIndex) {
	Path<double> path(dataPtr, dataDimension, dataLength);
	return path[lengthIndex][dimIndex];
}
