/* Provide an empty __gnu_cxx namespace on non-GCC compilers (Apple Clang). */
#ifndef COMPAT_GNU_CXX_H
#define COMPAT_GNU_CXX_H
#ifdef __cplusplus
#if !defined(__GNUC__) || defined(__clang__)
namespace __gnu_cxx {}
#endif
#endif
#endif
