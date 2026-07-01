
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sse_defs.h"

/* Determine Platform */
/*
#if defined(__x86_64__) || defined(_M_AMD64) || defined(_M_X64)
#define PLATFORM_X64
#else
#define PLATFORM_X86
#endif
*/

#ifdef INTEL_COMPATIBLE_HARDWARE

//
// Get cpuid
//
void cpuid(unsigned info, unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
  // NOTE: This code is modified from Wikipedia article: http://en.wikipedia.org/wiki/CPUID
  //       on 4/18/2014
    __asm__ __volatile__(
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
        "pushq %%rbx     \n\t" /* save %rbx */
#else
        "pushl %%ebx     \n\t" /* save %ebx */
#endif
        "cpuid            \n\t"
        "movl %%ebx ,%[ebx]  \n\t" /* write the result into output var */
        "movl %%ecx ,%[ecx]  \n\t" /* write the result into output var */
        "movl %%edx ,%[edx]  \n\t" /* write the result into output var */
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
        "popq %%rbx \n\t"
#else
        "popl %%ebx \n\t"
#endif
        : "=a"(*eax), [ebx] "=r"(*ebx), [ecx] "=c"(*ecx), [edx] "=d"(*edx)
        : "a"(info));
    /*
#ifdef PLATFORM_X64
  __asm__ (
	   "push %%rbx;"                // save ebx
	   "mov %1, %%eax; "            // info into eax
	   "cpuid;"
	   "mov %%edx, %0;"             // edx into *edx
	   "pop %%rbx;"
	   :"=d"(*edx)                     // output
	   :"r"(info)                      // input
	   :"%eax","%ebx","%ecx","%edx" // clobbered register
	   );
#else
  __asm__ (
	   "push %%ebx;"                // save ebx
	   "mov %1, %%eax; "            // info into eax
	   "cpuid;"
	   "mov %%edx, %0;"             // edx into *edx
	   "pop %%ebx;"
	   :"=r"(*edx)                     // output
	   :"r"(info)                      // input
	   :"%eax","ebx","%ecx","%edx" // clobbered register
	   );
#endif
    */

  return;
}
#endif

//
// Returns Vendor ID. Used to tell the difference between AMD and Intel
//
// 0 = unregocnized
// 1 = Intel
// 2 = AMD
//
void get_vendor_id(int *vendor_id) {
  *vendor_id = 0;
#ifdef INTEL_COMPATIBLE_HARDWARE
  unsigned eax;
  union vendor_t {
    char str[13]; // 12 chars + \0
    struct {
      unsigned ebx,edx,ecx;
    };
  };
  union vendor_t vendor;
  cpuid(0,&eax,&vendor.ebx,&vendor.ecx,&vendor.edx);
  vendor.str[12] = '\0';
  if (strcmp(vendor.str, "GenuineIntel") == 0) {
    *vendor_id = 1;
  } else if (strcmp(vendor.str, "AMDisbetter!") == 0 ||
	     strcmp(vendor.str, "AuthenticAMD") == 0) {
    *vendor_id = 2;
  }
#endif
}

//
// Returns SIMD version for Fortran
//
// simd_compile = compiling flags
// simd_cpu     = cpu flags
//
// 0 = no SSE
// 1 = SSE
// 2 = SSE2
// 3 = SSE3
// 4 = SSE4.1
// 5 = SSE4.2
// 6 = AVX
// 7 = AVX + FMA
// 8 = AVX2
// 9 = AVX2 + FMA
// 20 = MIC
//
void get_simd_version(int *simd_compile, int *simd_cpu) {
  unsigned eax,ebx,ecx,edx;

  *simd_compile = 0;
  
#ifndef __PGI
#ifdef INTEL_COMPATIBLE_HARDWARE

#ifdef __SSE__
  *simd_compile = 1;
#endif

#ifdef __SSE2__
  *simd_compile = 2;
#endif

#ifdef __SSE3__
  *simd_compile = 3;
#endif

#ifdef __AVX__
  *simd_compile = 6;
#endif

  // For MIC, expects the hardware to be there!
#ifdef __MIC__
  *simd_compile = 20;
  return;
#endif

  // Get Processor info and feature bits
  cpuid(1,&eax,&ebx,&ecx,&edx);
#endif //INTEL_COMPATIBLE_HARDWARE

  *simd_cpu = 0;

  /*
  printf("edx = %x ecx = %x\n",edx,ecx);
#ifdef PLATFORM_X64
  printf("PLATFORM_X64\n");
#else
  printf("no PLATFORM_X64\n");
#endif  
  */

#ifdef INTEL_COMPATIBLE_HARDWARE
  // SSE
  if (edx & (1 << 25)) {
    *simd_cpu = 1;
  }
  
  // SSE2
  if (edx & (1 << 26)) {
    *simd_cpu = 2;
  }

  // SSE3
  if (ecx & (1 << 9)) {
    *simd_cpu = 3;
  }

  // SSE4.1
  if ((ecx & (1 << 19)) != 0) {
    *simd_cpu = 4;
  }

  // SSE4.2
  if (ecx & (1 << 20)) {
    *simd_cpu = 5;
  }

  if (ecx & (1 << 28)) {
    // AVX
    *simd_cpu = 6;
    if (ebx & (1 << 5)) {
      // AVX2
      *simd_cpu = 8;
    }
    if (ecx & (1 << 12)) {
      // FMA
      (*simd_cpu)++;
    }
  }
#endif
#endif /* ! __PGI */
  return;
}
