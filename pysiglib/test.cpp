#include <Windows.h>
#include <iostream>
#include <vector>


int main()
{
    HMODULE h = ::LoadLibraryA("C:\\Users\\Shmelev\\source\\repos\\pySigLib\\pysiglib\\pysiglib\\cusig.dll");
    if (h == NULL) {
        // failed to load dll
        return 1;
    }

    using FN = void(__cdecl*)(const long);

    // void cusig_hello_world(const long x)

    FN fn = (FN)::GetProcAddress(h, "cusig_hello_world");
    if (fn == NULL) {
        // failed to get address of calculate function
        return 2;
    }

    double x = 1.5;
    int n = 10;
    std::vector<double> arr(n, 2.5);
    (fn)(7);

    std::cout << "the end\n";
    return 0;
}
