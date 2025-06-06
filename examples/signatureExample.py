import pysiglib
import numpy as np
import iisignature
import timeit

if __name__ == '__main__':

    batchSize = 100
    length = 1000
    dim = 5
    degree = 5

    X = np.random.uniform(size=(batchSize, length, dim)).astype("double")

    start = timeit.default_timer()
    sig = iisignature.sig(X, degree)
    end = timeit.default_timer()

    print(end - start)
    print(sig[0][:5])

    start = timeit.default_timer()
    sig = pysiglib.signature(X, degree)
    end = timeit.default_timer()
    print(end - start)
    print(sig[0][1:6])