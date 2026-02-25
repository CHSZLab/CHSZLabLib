/* Compatibility shims for non-GCC compilers (Apple Clang, MSVC). */
#ifndef COMPAT_GNU_CXX_H
#define COMPAT_GNU_CXX_H
#ifdef __cplusplus
#if !defined(__GNUC__) || defined(__clang__)
namespace __gnu_cxx {}
/* KaMIS uses __gnu_parallel::partial_sum from <parallel/numeric>.
   Redirect to std::partial_sum on non-GCC compilers. */
#include <numeric>
namespace __gnu_parallel {
    using std::partial_sum;
}
#endif
#endif
#endif
