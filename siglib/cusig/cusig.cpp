#include "cupch.h"
#include <iostream>
#include <string>
#include "vectoradd.h"
#include "cusig.h"
#include <cassert>

void cusig_hello_world(const long x)
{
    // Array size of 2^16 (65536 elements)
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data
    std::vector<int> a;
    a.reserve(N);
    std::vector<int> b;
    b.reserve(N);
    std::vector<int> c;
    c.reserve(N);
    std::vector<int> d;
    d.reserve(N);

    // Initialize random numbers in each array
    for (int i = 0; i < N; i++) {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    //True value d
    for (int i = 0; i < N; i++) {
        d[i] = a[i] + b[i];
    }

    vectorAddCUDA(a, b, c, N);

    assert(std::equal(c.begin(), c.end(), d.begin()));

	std::cout << "cusig Hello World " + std::to_string(x) << std::endl;
}