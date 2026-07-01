#ifdef __SSE2__
#include <emmintrin.h>
#endif
#if defined(__INTEL_COMPILER) && !defined(__SSE3__)
#define __SSE3__
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


#ifdef __PGI
#warning "PGI compiler not supported for SSE intrinsics. SSE intrinsics switched off."
#undef __SSE2__
#undef __SSE3__
#undef __SSE4_1__
#undef __SSE4_2__
#undef __AVX__
#endif

#ifdef __INTEL_COMPILER
#define inline __forceinline
#endif

//Detect Intel compatible hardware. Is there a better way?
#undef INTEL_COMPATIBLE_HARDWARE
#if defined(__x86_64) || defined (__x86_64__) || defined(__MMX__) || defined(__amd64)
#define INTEL_COMPATIBLE_HARDWARE
#endif

#if defined(__ppc__) || defined(__ppc64__)
#undef INTEL_COMPATIBLE_HARDWARE
#endif
