/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "H5private.h"

H5_GCC_DIAG_OFF("larger-than=")
H5_CLANG_DIAG_OFF("overlength-strings")

/* clang-format off */
const char H5build_settings[]=
    "        SUMMARY OF THE HDF5 CONFIGURATION\n"
    "        =================================\n"
    "\n"
    "General Information:\n"
    "-------------------\n"
    "                   HDF5 Version: 1.14.6\n"
    "                  Configured on: Fri Oct 31 14:49:21 CET 2025\n"
    "                  Configured by: ericb@eric-pc\n"
    "                    Host system: x86_64-conda-linux-gnu\n"
    "              Uname information: Linux eric-pc 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux\n"
    "                       Byte sex: little-endian\n"
    "             Installation point: /home/ericb/anaconda3/envs/molpro-build\n"
    "\n"
    "Compiling Options:\n"
    "------------------\n"
    "                     Build Mode: production\n"
    "              Debugging Symbols: no\n"
    "                        Asserts: no\n"
    "                      Profiling: no\n"
    "             Optimization Level: high\n"
    "\n"
    "Linking Options:\n"
    "----------------\n"
    "                      Libraries: static, shared\n"
    "  Statically Linked Executables: \n"
    "                        LDFLAGS: -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,-rpath,/home/ericb/anaconda3/envs/molpro-build/lib -Wl,-rpath-link,/home/ericb/anaconda3/envs/molpro-build/lib -L/home/ericb/anaconda3/envs/molpro-build/lib\n"
    "                     H5_LDFLAGS: \n"
    "                     AM_LDFLAGS: \n"
    "                Extra libraries: -lz -ldl -lm \n"
    "                       Archiver: /home/ericb/anaconda3/envs/molpro-build/bin/x86_64-conda-linux-gnu-ar\n"
    "                       AR_FLAGS: cr\n"
    "                         Ranlib: /home/ericb/anaconda3/envs/molpro-build/bin/x86_64-conda-linux-gnu-ranlib\n"
    "\n"
    "Languages:\n"
    "----------\n"
    "                              C: yes\n"
    "                     C Compiler: /home/ericb/anaconda3/envs/molpro-build/bin/mpicc ( MPICH version 4.1.1 Configured with: /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/configure --build=x86_64-build_conda-linux-gnu --host=x86_64-build_conda-linux-gnu --target=x86_64-conda-linux-gnu --prefix=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc_built --exec_prefix=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc_built --with-sysroot=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc_built/x86_64-conda-linux-gnu/sysroot --with-local-prefix=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc_built/x86_64-conda-linux-gnu/sysroot --enable-long-long --disable-multilib --disable-nls --with-gmp=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/buildtools --with-mpfr=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/buildtools --with-mpc=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/buildtools --with-isl=/opt/conda/conda-bld/gcc-compiler_1654084175708/work/buildtools --without-zstd --enable-languages=c,c++,fortran,objc,obj-c++ --enable-__cxa_atexit --disable-libmudflap --enable-libgomp --enable-libquadmath --enable-libquadmath-support --enable-libsanitizer --disable-libstdcxx-verbose --enable-lto --enable-libcc1 --enable-threads=posix --enable-plugin --with-pkgversion='Anaconda gcc' --enable-libmpx built with gcc version 11.2.0 (Anaconda gcc))\n"
    "                       CPPFLAGS: -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/ericb/anaconda3/envs/molpro-build/include\n"
    "                    H5_CPPFLAGS: -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L   -DNDEBUG -UH5_DEBUG_API -I/home/ericb/mmml/hdf5-1.14.6/src/H5FDsubfiling\n"
    "                    AM_CPPFLAGS: \n"
    "                        C Flags: -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/ericb/anaconda3/envs/molpro-build/include\n"
    "                     H5 C Flags:  -std=c99  -Wall -Wcast-qual -Wconversion -Wextra -Wfloat-equal -Wformat=2 -Winit-self -Winvalid-pch -Wmissing-include-dirs -Wshadow -Wundef -Wwrite-strings -pedantic -Wno-c++-compat -Wlarger-than=2560 -Wlogical-op -Wframe-larger-than=16384 -Wpacked-bitfield-compat -Wsync-nand -Wno-unsuffixed-float-constants -Wdouble-promotion -Wtrampolines -Wstack-usage=8192 -Wmaybe-uninitialized -Wdate-time -Warray-bounds=2 -Wc99-c11-compat -Wduplicated-cond -Whsa -Wnormalized -Wnull-dereference -Wunused-const-variable -Walloca -Walloc-zero -Wduplicated-branches -Wformat-overflow=2 -Wformat-truncation=1 -Wattribute-alias -Wshift-overflow=2 -Wattribute-alias=2 -Wmissing-profile -Wc11-c2x-compat -fstdarg-opt -fdiagnostics-urls=never -fno-diagnostics-color -s  -Wbad-function-cast -Wcast-align -Wformat -Wimplicit-function-declaration -Wint-to-pointer-cast -Wmissing-declarations -Wmissing-prototypes -Wnested-externs -Wold-style-definition -Wpacked -Wpointer-sign -Wpointer-to-int-cast -Wredundant-decls -Wstrict-prototypes -Wswitch -Wunused-but-set-variable -Wunused-variable -Wunused-function -Wunused-parameter -Wincompatible-pointer-types -Wint-conversion -Wshadow -Wrestrict -Wcast-function-type -Wmaybe-uninitialized -Wcast-align=strict -Wno-aggregate-return -Wno-inline -Wno-missing-format-attribute -Wno-missing-noreturn -Wno-overlength-strings -Wno-jump-misses-init -Wstrict-overflow=2 -Wno-suggest-attribute=const -Wno-suggest-attribute=noreturn -Wno-suggest-attribute=pure -Wno-suggest-attribute=format -Wno-suggest-attribute=cold -Wno-suggest-attribute=malloc -O3\n"
    "                     AM C Flags: \n"
    "               Shared C Library: yes\n"
    "               Static C Library: yes\n"
    "\n"
    "\n"
    "                        Fortran: no\n"
    "\n"
    "                            C++: no\n"
    "\n"
    "                           Java: no\n"
    "\n"
    "\n"
    "Features:\n"
    "---------\n"
    "                     Parallel HDF5: yes\n"
    "  Parallel Filtered Dataset Writes: yes\n"
    "                Large Parallel I/O: yes\n"
    "                High-level library: yes\n"
    "Dimension scales w/ new references: no\n"
    "                  Build HDF5 Tests: yes\n"
    "                  Build HDF5 Tools: yes\n"
    "                   Build GIF Tools: no\n"
    "                      Threadsafety: no\n"
    "               Default API mapping: v114\n"
    "    With deprecated public symbols: yes\n"
    "            I/O filters (external): deflate(zlib)\n"
    "                  _Float16 support: no\n"
    "                     Map (H5M) API: no\n"
    "                        Direct VFD: no\n"
    "                        Mirror VFD: no\n"
    "                     Subfiling VFD: no\n"
    "                (Read-Only) S3 VFD: no\n"
    "              (Read-Only) HDFS VFD: no\n"
    "    Packages w/ extra debug output: none\n"
    "                       API tracing: no\n"
    "              Using memory checker: no\n"
    "                  Use file locking: best-effort\n"
    "         Strict file format checks: no\n"
    "      Optimization instrumentation: no\n"
;
/* clang-format on */

H5_GCC_DIAG_ON("larger-than=")
H5_CLANG_DIAG_OFF("overlength-strings")
