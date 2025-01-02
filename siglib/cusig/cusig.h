#pragma once

#ifdef CUSIG_EXPORTS
#define CUSIG_API __declspec(dllexport)
#else
#define CUSIG_API __declspec(dllimport)
#endif

extern "C" CUSIG_API void cusig_hello_world(const long x);