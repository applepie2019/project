/******************************************************************************
** Copyright (c) 2013-2021, Alexander Heinecke                               **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/

#if 0
#define USE_CORE_PERF_SNP
#endif
#if 0
#define USE_CORE_PERF_L2IN
#endif
#if 0
#define USE_CORE_PERF_L3HITS
#endif
#if 0
#define USE_CORE_PERF_IPC
#endif
#if 0
#define USE_UNCORE_PERF_DRAM_BW
#endif
#if 0
#define USE_UNCORE_PERF_LLC_VICTIMS
#endif
#if 0
#define USE_UNCORE_PERF_CHA_UTIL
#endif
#if 0
#define USE_UNCORE_PREF_AK_UTIL
#endif
#if 0
#define USE_UNCORE_PREF_IV_UTIL
#endif
#if 0
#define USE_UNCORE_PREF_CHA_XSNP_RESP
#endif
#if 0
#define USE_UNCORE_PREF_CHA_CORE_SNP
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <immintrin.h>

#define MY_MIN(A,B) (((A)<(B))?(A):(B))

#if 0
#define USE_ROTATION_SCHEME
#endif
#if 0
#define USE_2LROTATION_SCHEME
#endif
#if 0
#define CLFLUSH_BUFFER_WHEN_DONE
#endif
#if 0
#define PERFORM_DETAILED_ANALYSIS
#endif
#if 0
#define USE_CLDEMOTE
#endif

#if defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_IPC) || defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL) || defined(USE_UNCORE_PREF_CHA_XSNP_RESP) || defined(USE_UNCORE_PREF_CHA_CORE_SNP) || defined(USE_CORE_PERF_L3HITS)
#  include "../common/perf_counter_markers.h"
#endif

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

void read_buffer( char* i_buffer, size_t i_length ) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $2048, %%r9\n\t"
                       "vmovapd     0(%%r8),   %%zmm0\n\t"
                       "vmovapd    64(%%r8),   %%zmm1\n\t"
                       "vmovapd   128(%%r8),   %%zmm2\n\t"
                       "vmovapd   192(%%r8),   %%zmm3\n\t"
                       "vmovapd   256(%%r8),   %%zmm4\n\t"
                       "vmovapd   320(%%r8),   %%zmm5\n\t"
                       "vmovapd   384(%%r8),   %%zmm6\n\t"
                       "vmovapd   448(%%r8),   %%zmm7\n\t"
                       "vmovapd   512(%%r8),   %%zmm8\n\t"
                       "vmovapd   576(%%r8),   %%zmm9\n\t"
                       "vmovapd   640(%%r8),  %%zmm10\n\t"
                       "vmovapd   704(%%r8),  %%zmm11\n\t"
                       "vmovapd   768(%%r8),  %%zmm12\n\t"
                       "vmovapd   832(%%r8),  %%zmm13\n\t"
                       "vmovapd   896(%%r8),  %%zmm14\n\t"
                       "vmovapd   960(%%r8),  %%zmm15\n\t"
                       "vmovapd  1024(%%r8),  %%zmm16\n\t"
                       "vmovapd  1088(%%r8),  %%zmm17\n\t"
                       "vmovapd  1152(%%r8),  %%zmm18\n\t"
                       "vmovapd  1216(%%r8),  %%zmm19\n\t"
                       "vmovapd  1280(%%r8),  %%zmm20\n\t"
                       "vmovapd  1344(%%r8),  %%zmm21\n\t"
                       "vmovapd  1408(%%r8),  %%zmm22\n\t"
                       "vmovapd  1472(%%r8),  %%zmm23\n\t"
                       "vmovapd  1536(%%r8),  %%zmm24\n\t"
                       "vmovapd  1600(%%r8),  %%zmm25\n\t"
                       "vmovapd  1664(%%r8),  %%zmm26\n\t"
                       "vmovapd  1728(%%r8),  %%zmm27\n\t"
                       "vmovapd  1792(%%r8),  %%zmm28\n\t"
                       "vmovapd  1856(%%r8),  %%zmm29\n\t"
                       "vmovapd  1920(%%r8),  %%zmm30\n\t"
                       "vmovapd  1984(%%r8),  %%zmm31\n\t"
                       "addq $2048, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#elif __AVX__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $512, %%r9\n\t"
                       "vmovapd    0(%%r8),   %%ymm0\n\t"
                       "vmovapd   32(%%r8),   %%ymm1\n\t"
                       "vmovapd   64(%%r8),   %%ymm2\n\t"
                       "vmovapd   96(%%r8),   %%ymm3\n\t"
                       "vmovapd  128(%%r8),   %%ymm4\n\t"
                       "vmovapd  160(%%r8),   %%ymm5\n\t"
                       "vmovapd  192(%%r8),   %%ymm6\n\t"
                       "vmovapd  224(%%r8),   %%ymm7\n\t"
                       "vmovapd  256(%%r8),   %%ymm8\n\t"
                       "vmovapd  288(%%r8),   %%ymm9\n\t"
                       "vmovapd  320(%%r8),  %%ymm10\n\t"
                       "vmovapd  352(%%r8),  %%ymm11\n\t"
                       "vmovapd  384(%%r8),  %%ymm12\n\t"
                       "vmovapd  416(%%r8),  %%ymm13\n\t"
                       "vmovapd  448(%%r8),  %%ymm14\n\t"
                       "vmovapd  480(%%r8),  %%ymm15\n\t"
                       "addq $512, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#elif __SSE2__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $256, %%r9\n\t"
                       "movapd    0(%%r8),   %%xmm0\n\t"
                       "movapd   16(%%r8),   %%xmm1\n\t"
                       "movapd   32(%%r8),   %%xmm2\n\t"
                       "movapd   48(%%r8),   %%xmm3\n\t"
                       "movapd   64(%%r8),   %%xmm4\n\t"
                       "movapd   80(%%r8),   %%xmm5\n\t"
                       "movapd   96(%%r8),   %%xmm6\n\t"
                       "movapd  112(%%r8),   %%xmm7\n\t"
                       "movapd  128(%%r8),   %%xmm8\n\t"
                       "movapd  144(%%r8),   %%xmm9\n\t"
                       "movapd  160(%%r8),  %%xmm10\n\t"
                       "movapd  176(%%r8),  %%xmm11\n\t"
                       "movapd  192(%%r8),  %%xmm12\n\t"
                       "movapd  208(%%r8),  %%xmm13\n\t"
                       "movapd  224(%%r8),  %%xmm14\n\t"
                       "movapd  240(%%r8),  %%xmm15\n\t"
                       "addq $256, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#else
#error need at least SSE2
#endif
}

void read_buffer_128B_odd( char* i_buffer) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "vmovapd     0(%%r8),   %%zmm0\n\t"
                       "vmovapd    64(%%r8),   %%zmm1\n\t"
                       : : "m"(i_buffer) : "r8","zmm0","zmm1");
#else
#error need at least SSE2
#endif
}

void read_buffer_128B_even( char* i_buffer) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%r9\n\t"
                       "vmovapd     0(%%r9),   %%zmm2\n\t"
                       "vmovapd    64(%%r9),   %%zmm3\n\t"
                       : : "m"(i_buffer) : "r9","zmm2","zmm3");
#else
#error need at least SSE2
#endif
}

void read_buffer_256B( char* i_buffer) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "vmovapd     0(%%r8),   %%zmm0\n\t"
                       "vmovapd    64(%%r8),   %%zmm1\n\t"
                       "vmovapd   128(%%r8),   %%zmm2\n\t"
                       "vmovapd   192(%%r8),   %%zmm3\n\t"
                       : : "m"(i_buffer) : "r8","zmm0","zmm1","zmm2","zmm3");
#else
#error need at least SSE2
#endif
}

void read_cldemote_buffer( char* i_buffer, size_t i_length ) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $2048, %%r9\n\t"
                       "vmovapd     0(%%r8),   %%zmm0\n\t"
                       "vmovapd    64(%%r8),   %%zmm1\n\t"
                       "vmovapd   128(%%r8),   %%zmm2\n\t"
                       "vmovapd   192(%%r8),   %%zmm3\n\t"
                       "vmovapd   256(%%r8),   %%zmm4\n\t"
                       "vmovapd   320(%%r8),   %%zmm5\n\t"
                       "vmovapd   384(%%r8),   %%zmm6\n\t"
                       "vmovapd   448(%%r8),   %%zmm7\n\t"
                       "vmovapd   512(%%r8),   %%zmm8\n\t"
                       "vmovapd   576(%%r8),   %%zmm9\n\t"
                       "vmovapd   640(%%r8),  %%zmm10\n\t"
                       "vmovapd   704(%%r8),  %%zmm11\n\t"
                       "vmovapd   768(%%r8),  %%zmm12\n\t"
                       "vmovapd   832(%%r8),  %%zmm13\n\t"
                       "vmovapd   896(%%r8),  %%zmm14\n\t"
                       "vmovapd   960(%%r8),  %%zmm15\n\t"
                       "vmovapd  1024(%%r8),  %%zmm16\n\t"
                       "vmovapd  1088(%%r8),  %%zmm17\n\t"
                       "vmovapd  1152(%%r8),  %%zmm18\n\t"
                       "vmovapd  1216(%%r8),  %%zmm19\n\t"
                       "vmovapd  1280(%%r8),  %%zmm20\n\t"
                       "vmovapd  1344(%%r8),  %%zmm21\n\t"
                       "vmovapd  1408(%%r8),  %%zmm22\n\t"
                       "vmovapd  1472(%%r8),  %%zmm23\n\t"
                       "vmovapd  1536(%%r8),  %%zmm24\n\t"
                       "vmovapd  1600(%%r8),  %%zmm25\n\t"
                       "vmovapd  1664(%%r8),  %%zmm26\n\t"
                       "vmovapd  1728(%%r8),  %%zmm27\n\t"
                       "vmovapd  1792(%%r8),  %%zmm28\n\t"
                       "vmovapd  1856(%%r8),  %%zmm29\n\t"
                       "vmovapd  1920(%%r8),  %%zmm30\n\t"
                       "vmovapd  1984(%%r8),  %%zmm31\n\t"
                       "cldemote     0(%%r8)\n\t"
                       "cldemote    64(%%r8)\n\t"
                       "cldemote   128(%%r8)\n\t"
                       "cldemote   192(%%r8)\n\t"
                       "cldemote   256(%%r8)\n\t"
                       "cldemote   320(%%r8)\n\t"
                       "cldemote   384(%%r8)\n\t"
                       "cldemote   448(%%r8)\n\t"
                       "cldemote   512(%%r8)\n\t"
                       "cldemote   576(%%r8)\n\t"
                       "cldemote   640(%%r8)\n\t"
                       "cldemote   704(%%r8)\n\t"
                       "cldemote   768(%%r8)\n\t"
                       "cldemote   832(%%r8)\n\t"
                       "cldemote   896(%%r8)\n\t"
                       "cldemote   960(%%r8)\n\t"
                       "cldemote  1024(%%r8)\n\t"
                       "cldemote  1088(%%r8)\n\t"
                       "cldemote  1152(%%r8)\n\t"
                       "cldemote  1216(%%r8)\n\t"
                       "cldemote  1280(%%r8)\n\t"
                       "cldemote  1344(%%r8)\n\t"
                       "cldemote  1408(%%r8)\n\t"
                       "cldemote  1472(%%r8)\n\t"
                       "cldemote  1536(%%r8)\n\t"
                       "cldemote  1600(%%r8)\n\t"
                       "cldemote  1664(%%r8)\n\t"
                       "cldemote  1728(%%r8)\n\t"
                       "cldemote  1792(%%r8)\n\t"
                       "cldemote  1856(%%r8)\n\t"
                       "cldemote  1920(%%r8)\n\t"
                       "cldemote  1984(%%r8)\n\t"
                       "addq $2048, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm16","xmm17","xmm18","xmm19","xmm20","xmm21","xmm22","xmm23","xmm24","xmm25","xmm26","xmm27","xmm28","xmm29","xmm30","xmm31");
#elif __AVX__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $512, %%r9\n\t"
                       "vmovapd    0(%%r8),   %%ymm0\n\t"
                       "vmovapd   32(%%r8),   %%ymm1\n\t"
                       "vmovapd   64(%%r8),   %%ymm2\n\t"
                       "vmovapd   96(%%r8),   %%ymm3\n\t"
                       "vmovapd  128(%%r8),   %%ymm4\n\t"
                       "vmovapd  160(%%r8),   %%ymm5\n\t"
                       "vmovapd  192(%%r8),   %%ymm6\n\t"
                       "vmovapd  224(%%r8),   %%ymm7\n\t"
                       "vmovapd  256(%%r8),   %%ymm8\n\t"
                       "vmovapd  288(%%r8),   %%ymm9\n\t"
                       "vmovapd  320(%%r8),  %%ymm10\n\t"
                       "vmovapd  352(%%r8),  %%ymm11\n\t"
                       "vmovapd  384(%%r8),  %%ymm12\n\t"
                       "vmovapd  416(%%r8),  %%ymm13\n\t"
                       "vmovapd  448(%%r8),  %%ymm14\n\t"
                       "vmovapd  480(%%r8),  %%ymm15\n\t"
                       "addq $512, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#elif __SSE2__
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $256, %%r9\n\t"
                       "movapd    0(%%r8),   %%xmm0\n\t"
                       "movapd   16(%%r8),   %%xmm1\n\t"
                       "movapd   32(%%r8),   %%xmm2\n\t"
                       "movapd   48(%%r8),   %%xmm3\n\t"
                       "movapd   64(%%r8),   %%xmm4\n\t"
                       "movapd   80(%%r8),   %%xmm5\n\t"
                       "movapd   96(%%r8),   %%xmm6\n\t"
                       "movapd  112(%%r8),   %%xmm7\n\t"
                       "movapd  128(%%r8),   %%xmm8\n\t"
                       "movapd  144(%%r8),   %%xmm9\n\t"
                       "movapd  160(%%r8),  %%xmm10\n\t"
                       "movapd  176(%%r8),  %%xmm11\n\t"
                       "movapd  192(%%r8),  %%xmm12\n\t"
                       "movapd  208(%%r8),  %%xmm13\n\t"
                       "movapd  224(%%r8),  %%xmm14\n\t"
                       "movapd  240(%%r8),  %%xmm15\n\t"
                       "addq $256, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");
#else
#error need at least SSE2
#endif
}

void clflush_buffer( char* i_buffer, size_t i_length ) {
  __asm__ __volatile__("movq %0, %%r8\n\t"
                       "movq %1, %%r9\n\t"
                       "1:\n\t"
                       "subq $2048, %%r9\n\t"
                       "clflushopt     0(%%r8)\n\t"
                       "clflushopt    64(%%r8)\n\t"
                       "clflushopt   128(%%r8)\n\t"
                       "clflushopt   192(%%r8)\n\t"
                       "clflushopt   256(%%r8)\n\t"
                       "clflushopt   320(%%r8)\n\t"
                       "clflushopt   384(%%r8)\n\t"
                       "clflushopt   448(%%r8)\n\t"
                       "clflushopt   512(%%r8)\n\t"
                       "clflushopt   576(%%r8)\n\t"
                       "clflushopt   640(%%r8)\n\t"
                       "clflushopt   704(%%r8)\n\t"
                       "clflushopt   768(%%r8)\n\t"
                       "clflushopt   832(%%r8)\n\t"
                       "clflushopt   896(%%r8)\n\t"
                       "clflushopt   960(%%r8)\n\t"
                       "clflushopt  1024(%%r8)\n\t"
                       "clflushopt  1088(%%r8)\n\t"
                       "clflushopt  1152(%%r8)\n\t"
                       "clflushopt  1216(%%r8)\n\t"
                       "clflushopt  1280(%%r8)\n\t"
                       "clflushopt  1344(%%r8)\n\t"
                       "clflushopt  1408(%%r8)\n\t"
                       "clflushopt  1472(%%r8)\n\t"
                       "clflushopt  1536(%%r8)\n\t"
                       "clflushopt  1600(%%r8)\n\t"
                       "clflushopt  1664(%%r8)\n\t"
                       "clflushopt  1728(%%r8)\n\t"
                       "clflushopt  1792(%%r8)\n\t"
                       "clflushopt  1856(%%r8)\n\t"
                       "clflushopt  1920(%%r8)\n\t"
                       "clflushopt  1984(%%r8)\n\t"
                       "addq $2048, %%r8\n\t"
                       "cmpq $0, %%r9\n\t"
                       "jg 1b\n\t"
                       : : "m"(i_buffer), "r"(i_length) : "r8","r9");
}

int main(int argc, char* argv[]) {
  size_t l_n_bytes = 0;
  size_t l_n_levels = 0;
  size_t l_n_parts = 0;
  size_t l_n_workers = 0;
  size_t l_n_oiters = 0;
  size_t l_n_iiters = 0;
  //char** l_n_buffers = 0;
  //char** l_n_buffers = 0;
  struct timeval l_startTime, l_endTime;
  struct timeval *l_startTime_arr, *l_endTime_arr;
  double *l_iterTimes;
  double l_avgtime;
  double l_mintime;
  double l_maxtime;
  double l_totalGiB, l_totalGiB_dram;
  size_t i, j, k;

#if defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_IPC) || defined(USE_CORE_PERF_L3HITS)
  ctrs_core cc_a, cc_b, cc_s;
  zero_core_ctrs( &cc_a );
  zero_core_ctrs( &cc_b );
  zero_core_ctrs( &cc_s );

#if defined(USE_CORE_PERF_L2IN)
  setup_core_ctrs(CTRS_EXP_L2_BW);
#endif
#if defined(USE_CORE_PERF_L3HITS)
  setup_core_ctrs(CTRS_EXP_L3_BW);
#endif
#if defined(USE_CORE_PERF_SNP)
  setup_core_ctrs(CTRS_EXP_CORE_SNP_RSP);
#endif
#if defined(USE_CORE_PERF_IPC)
  setup_core_ctrs(CTRS_EXP_IPC);
#endif
#endif
#if defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL)  || defined(USE_UNCORE_PREF_CHA_XSNP_RESP) || defined(USE_UNCORE_PREF_CHA_CORE_SNP) || defined(USE_CORE_PERF_L3HITS)
  ctrs_uncore uc_a, uc_b, uc_s;
  zero_uncore_ctrs( &uc_a );
  zero_uncore_ctrs( &uc_b );
  zero_uncore_ctrs( &uc_s );

#if defined(USE_UNCORE_PERF_DRAM_BW)
  setup_uncore_ctrs( CTRS_EXP_DRAM_CAS );
#endif
#if defined(USE_UNCORE_PERF_LLC_VICTIMS)
  setup_uncore_ctrs( CTRS_EXP_CHA_LLC_LOOKUP_VICTIMS );
#endif
#if defined(USE_UNCORE_PERF_CHA_UTIL)
  setup_uncore_ctrs( CTRS_EXP_CHA_UTIL );
#endif
#if defined(USE_UNCORE_PREF_AK_UTIL)
  setup_uncore_ctrs( CTRS_EXP_CMS_AK );
#endif
#if defined(USE_UNCORE_PREF_IV_UTIL)
  setup_uncore_ctrs( CTRS_EXP_CMS_IV );
#endif
#if defined(USE_UNCORE_PREF_CHA_XSNP_RESP)
  setup_uncore_ctrs(CTRS_EXP_CHA_XSNP_RESP);
#endif
#if defined(USE_UNCORE_PREF_CHA_CORE_SNP)
  setup_uncore_ctrs(CTRS_EXP_CHA_CORE_SNP);
#endif
#endif

  if ( argc != 7 ) {
    printf("Wrong parameters: ./app [# of 2K blocks] [#levels] [#partitions] [#workers] [#outer iterations] [#inner iterations]\n");
    return -1;
  }

  /* reading values from the command line */
  l_n_bytes = ((size_t)atoi(argv[1]))*2048;
  l_n_levels = atoi(argv[2]);
  l_n_parts = atoi(argv[3]);
  l_n_workers = atoi(argv[4]);
  l_n_oiters = atoi(argv[5]);
  l_n_iiters = atoi(argv[6]);

  /* validate the inputs */
  if ( (l_n_levels < 1) || (l_n_oiters < 1) || (l_n_iiters < 1) ) {
    printf("levels and iterations count needs to be non-zero and positive\n");
    return -1;
  }
  if ( (l_n_bytes % 2048) != 0 ) {
    printf("each partition needs to be at least 2KB and the size of each partition needs to be a multipe of 2KB. ABORT!\n");
    return -1;
  }
  if ( l_n_workers <= 1 ) {
    printf("num of workers must be > 1\n");
    return -1;
  }
#if 0
  if ( (l_n_bytes/2048) % (l_n_workers-1) !=0 ) {
    printf("the size of each partition needs to be a multipe of num_workers-1. ABORT!\n");
    return -1;
  }
#endif
  if ( (l_n_parts < 1) || ((l_n_workers % l_n_parts) != 0) ) {
    printf("paritions need to evenly divide workers. ABORT!\n");
    return -1;
  }

#ifdef __AVX512F__
  printf("using AVX512F\n");
#elif __AVX__
  printf("using AVX\n");
#elif __SSE2__
  printf("using SSE2\n");
#else
#error need at least SSE2
#endif
  char** l_n_buffers[l_n_workers-1];
  /* allocating data */
  for ( j = 0; j < l_n_workers - 1; j++) {
    l_n_buffers[j] = (char**) malloc( l_n_levels*sizeof(char**) );
    for ( i = 0; i < l_n_levels; ++i ) {
      posix_memalign( (void**)&(l_n_buffers[j][i]), 4096, l_n_bytes*sizeof(char) );
      memset( l_n_buffers[j][i], (int)i, l_n_bytes );
    }
  }
  l_startTime_arr = (struct timeval*) malloc( l_n_oiters*sizeof(struct timeval) );
  l_endTime_arr = (struct timeval*) malloc( l_n_oiters*sizeof(struct timeval) );
  l_iterTimes = (double*) malloc( l_n_oiters*sizeof(double) );
  l_totalGiB = ((double)(l_n_bytes * l_n_levels * l_n_iiters * (l_n_workers / l_n_parts)))/(1024.0*1024.0*1024);
  l_totalGiB_dram = ((double)(l_n_bytes * l_n_levels))/(1024.0*1024.0*1024);

  printf("#Levels                                 : %lld\n", l_n_levels );
  printf("#iterations per level                   : %lld\n", l_n_iiters );
  printf("Buffer Size in GiB per level            : %f\n", (double)l_n_bytes*(l_n_workers-1)/(1024.0*1024.0*1024.0));
  printf("Buffer Size in bytes per level          : %lld\n", l_n_bytes );
  printf("#workers used                           : %lld\n", l_n_workers );
  printf("#partition used                         : %lld\n", l_n_parts );
  printf("each workers reads #bytes                : %lld\n", (l_n_bytes) );
  printf("each worker reads %% of buffer           : %f\n", (((double)(l_n_bytes / l_n_parts)) / ((double)l_n_bytes))*100.0 );
  printf("access indices worker 0\n");
  for ( i = 0; i < 1; ++i ) {
    size_t my_size = l_n_bytes / l_n_parts;
    size_t my_offset = (size_t)i / ( l_n_workers / l_n_parts );
    printf("  worker %.3i                             : [ %lld , %lld [ ; length = %lld\n", i, my_offset * my_size, (my_offset * my_size) + my_size, my_size );
  }
  printf("Iteration Volume in GiB form LLC        : %f\n", l_totalGiB );
  printf("Iteration Volume in GiB from DRAM       : %f\n", l_totalGiB_dram );


  printf("\nRunning detailed timing for round-robin read ...\n");
  {
    size_t* l_tsc_timer;
    size_t l_iter_to_analyze = 130;  //130
    size_t l_level_to_analyze = 250; //250
    volatile size_t l_counter = 0;
    if ( l_iter_to_analyze >= l_n_oiters ) {
      printf(" iter to analyze is out of bounds!\n ");
      return -1;
    }
    if ( l_level_to_analyze >= l_n_levels ) {
      printf(" iter to analyze is out of bounds!\n ");
      return -1;
    }
    /* allocatopm pf timer arrary */
    l_tsc_timer = (size_t*) malloc( l_n_workers*l_n_levels*l_n_oiters*8*sizeof(size_t) );
    memset( (void*)l_tsc_timer, 0, l_n_workers*l_n_levels*l_n_oiters*8*sizeof(size_t) );


#if defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_IPC) || defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_L3HITS)
    read_core_ctrs( &cc_a );
#endif

#if defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL) || defined(USE_UNCORE_PREF_CHA_XSNP_RESP) || defined(USE_UNCORE_PREF_CHA_CORE_SNP)
  read_uncore_ctrs( &uc_a );
#endif

#if defined(_OPENMP)
# pragma omp parallel private(i,j,k) num_threads(l_n_workers)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    for ( i = 0; i < l_n_oiters; ++i ) {
      for ( j = 0; j < l_n_levels; ++j ) {
        for ( k = 0; k < l_n_iiters; ++k ) {
          if (tid > 0) {
            char* my_buffer = l_n_buffers[tid-1][j];
            read_buffer( my_buffer, l_n_bytes);
	  }
#if defined(_OPENMP)
# pragma omp barrier
#endif
	  if (tid == 0) {
	    uint64_t counter = 0;
            l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0] = __rdtsc();
	    for (size_t offset = 0; offset < l_n_bytes; offset += 128) {
              for (size_t m = 0; m < l_n_workers-1; m++ ) {
                char* my_buffer = &l_n_buffers[m][j][offset];
#if 1
		if (counter & 1) {
#ifdef __AVX512F__
                __asm__ __volatile__("movq %0, %%r8\n\t"
                       "vmovapd     0(%%r8),   %%zmm0\n\t"
                       "vmovapd    64(%%r8),   %%zmm1\n\t"
		       ::"m"(my_buffer) : "r8","zmm0","zmm1");
#else
#error need at least SSE2
#endif
		} else {
#ifdef __AVX512F__
                __asm__ __volatile__("movq %0, %%r9\n\t"
                       "vmovapd     0(%%r9),   %%zmm2\n\t"
                       "vmovapd    64(%%r9),   %%zmm3\n\t"
		       ::"m"(my_buffer) : "r9","zmm2","zmm3");
#else
#error need at least SSE2
#endif
	        }
		counter++;
#endif

#if 0
	        read_buffer_256B(my_buffer);
#endif
              }
	    }
            l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1] = __rdtsc();
	  }
#if 1
#if defined(_OPENMP)
# pragma omp barrier
#endif
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2] = __rdtsc();
          for (size_t m = 0; m < l_n_workers-1; m++ )
            clflush_buffer(l_n_buffers[m][j], l_n_bytes);
          l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] = __rdtsc();
#endif
#if defined(_OPENMP)
# pragma omp barrier
#endif
        }
      }
    }
  }
#if defined(USE_CORE_PERF_SNP) || defined(USE_CORE_PERF_IPC) || defined(USE_CORE_PERF_L2IN) || defined(USE_CORE_PERF_L3HITS)
  read_core_ctrs( &cc_b );
  difa_core_ctrs( &cc_a, &cc_b, &cc_s );
  divi_core_ctrs( &cc_s, 1);
#endif
#if defined(USE_UNCORE_PERF_DRAM_BW) || defined(USE_UNCORE_PERF_LLC_VICTIMS) || defined(USE_UNCORE_PERF_CHA_UTIL) || defined(USE_UNCORE_PREF_AK_UTIL) || defined(USE_UNCORE_PREF_IV_UTIL) || defined(USE_UNCORE_PREF_CHA_XSNP_RESP) || defined(USE_UNCORE_PREF_CHA_CORE_SNP)
  read_uncore_ctrs( &uc_b );
  difa_uncore_ctrs( &uc_a, &uc_b, &uc_s );
  divi_uncore_ctrs( &uc_s, 1 );
#endif
    /* let's print the stats */
    {
      size_t l_tot_avg_cycles = 0;
      size_t l_tot_min_cycles = 0;
      size_t l_tot_max_cycles = 0;
      size_t l_avg_cycles = 0;
      size_t l_min_cycles = 0xffffffffffffffff;
      size_t l_max_cycles = 0;
      size_t l_load_bytes = (l_n_workers-1)*l_n_bytes;
      j = l_level_to_analyze;
      i = l_iter_to_analyze;
      size_t my_kern_size = l_n_bytes;
      size_t tid = 0;

      printf("\nLayer %lld and iteration %lld\n", j, i);
      printf("\nPhase II Perf - reading in data\n");
      printf("  per core:\n");
      for ( tid = 0; tid < 1; ++tid ) {
        size_t l_cycles = l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1] - l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0];
        l_avg_cycles += l_cycles;
        l_min_cycles = (l_cycles < l_min_cycles) ? l_cycles : l_min_cycles;
        l_max_cycles = (l_cycles > l_max_cycles) ? l_cycles : l_max_cycles;
        printf("Phase II worker %.3i: %f B/c (%lld, %lld, %lld, %lld) \n", tid, (double)l_load_bytes/(double)l_cycles, l_load_bytes, l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 0],  l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 1], l_cycles );
      }
      printf("  avg: %f, min: %f, max: %f B/c\n  avg: %lld, min: %lld, max: %lld cycles\n  vol: %lld bytes\n", (double)l_load_bytes/(double)l_avg_cycles, (double)l_load_bytes/(double)l_max_cycles, (double)l_load_bytes/(double)l_min_cycles,  l_avg_cycles, l_min_cycles, l_max_cycles, l_load_bytes);

      l_tot_avg_cycles += l_avg_cycles;
      l_tot_min_cycles += l_min_cycles;
      l_tot_max_cycles += l_max_cycles;

      l_avg_cycles = 0;
      l_min_cycles = 0xffffffffffffffff;
      l_max_cycles = 0;
      printf("\nPhase III Perf - flush caches\n");
      printf("  per core:\n");
      for ( tid = 0; tid < l_n_workers; ++tid ) {
        size_t l_cycles = l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3] - l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2];
        l_avg_cycles += l_cycles;
        l_min_cycles = (l_cycles < l_min_cycles) ? l_cycles : l_min_cycles;
        l_max_cycles = (l_cycles > l_max_cycles) ? l_cycles : l_max_cycles;
        printf("     worker %.3i: %f B/c (%lld, %lld, %lld, %lld) \n", tid, (double)l_load_bytes/(double)l_cycles, l_load_bytes, l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 3],  l_tsc_timer[(tid*l_n_levels*l_n_oiters*8) + (j*l_n_oiters*8) + (i*8) + 2], l_cycles );
        }
      l_avg_cycles /= l_n_workers;
      printf("  avg: %f, min: %f, max: %f B/c\n  avg: %lld, min: %lld, max: %lld cycles\n  vol: %lld bytes\n", (double)l_load_bytes/(double)l_avg_cycles, (double)l_load_bytes/(double)l_max_cycles, (double)l_load_bytes/(double)l_min_cycles, l_avg_cycles, l_min_cycles, l_max_cycles, l_load_bytes );

    }
#if defined(USE_CORE_PERF_SNP)
    {
      snp_rsp rsp;
      get_snp_rsp_core_ctrs( &cc_s, &rsp );
      printf("average #cycles per iteration : %f\n", rsp.cyc );
      printf("SNOOP RESP IHITI              : %f\n", rsp.ihiti );
      printf("SNOOP RESP IHITFSE            : %f\n", rsp.ihitfse );
      printf("SNOOP RESP IFWDM              : %f\n", rsp.ifwdm );
      printf("SNOOP RESP IFWDFE             : %f\n", rsp.ifwdfe );
      printf("avg SNOOP RESP IHITI / cycle  : %f\n", rsp.ihiti/rsp.cyc );
      printf("avg SNOOP RESP IHITFSE / cycle: %f\n", rsp.ihitfse/rsp.cyc );
      printf("avg SNOOP RESP IFWDM / cycle  : %f\n", rsp.ifwdm/rsp.cyc );
      printf("avg SNOOP RESP IFWDFE / cycle : %f\n", rsp.ifwdfe/rsp.cyc );
      for ( int core = 0; core < CTRS_NCORE; ++core ) {
        printf("core %d, #ihitfse %lld \n", core,  cc_s.core_snp_rsp_ihitfse[core]);
        printf("core %d, #ifwdfe %lld \n", core,  cc_s.core_snp_rsp_ifwdfe[core]);
        printf("core %d, #shitfse %lld \n", core,  cc_s.core_snp_rsp_shitfse[core]);
        printf("core %d, #sfwdfe %lld \n", core,  cc_s.core_snp_rsp_sfwdfe[core]);
      }
    }
#endif

#if defined(USE_CORE_PERF_IPC)
    {
      ipc_rate ipc;
      get_ipc_core_ctr( &cc_s, &ipc );
      printf("average #cycles per iteration : %f\n", ipc.cyc );
      printf("#intrs/core per iteration     : %f\n", ipc.instrs_core );
      printf("total #instr per iteration    : %f\n", ipc.instrs );
      printf("IPC per core                  : %f\n", ipc.ipc_core );
      printf("IPC per SOC                   : %f\n", ipc.ipc );
      for ( int core = 0; core < CTRS_NCORE; ++core )
        printf("core %d, #insts %d \n", core,  cc_s.instrs[core]);
  }
#endif
#if defined(USE_CORE_PERF_L2IN)
    {
      size_t l_avg_cycles = l_tsc_timer[(j*l_n_oiters*8) + (i*8) + 1] - l_tsc_timer[(j*l_n_oiters*8) + (i*8) + 0];
      bw_gibs bw_avg;
      get_l2_bw_core_ctrs( &cc_s, l_avg_cycles, &bw_avg );
      printf("AVG GiB/s (IN    L2): %f\n", bw_avg.rd);
      printf("AVG GiB/s (OUTS  L2): %f\n", bw_avg.wr);
      printf("AVG GiB/s (OUTNS L2): %f\n", bw_avg.wr2);
      printf("AVG GiB/s (DEM   L2): %f\n", bw_avg.wr3);
      printf("AVG GiB/s (DROP  L2): %f\n", bw_avg.wr4);
      for ( int core = 0; core < CTRS_NCORE; ++core )
        printf("core %d, #l2_lines %lld \n", core,  cc_s.l2_lines_in[core]);
    }
#endif
#if defined(USE_UNCORE_PREF_CHA_XSNP_RESP)
  {
    int cha = 0;
    for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
      printf("CHA %i: CMS cyc: %lld, xsnp_resp: %lld \n", cha, uc_s.cms_clockticks[cha], uc_s.xsnp_resp[cha]);
    }
  }
#endif
#if defined(USE_UNCORE_PREF_CHA_CORE_SNP)
  {
    int cha = 0;
    for ( cha = 0; cha < CTRS_NCHA; ++cha ) {
      printf("CHA %i: CMS cyc: %lld, core_snp: %lld \n", cha, uc_s.cms_clockticks[cha], uc_s.core_snp[cha]);
    }
  }
#endif
#if defined(USE_CORE_PERF_L3HITS)
  {
    for ( int core = 0; core < CTRS_NCORE; ++core ) {
      printf("core %d, #l3_hits %lld \n", core,  cc_s.l3_hits_xsnp_none[core]);
      printf("core %d, #l3_hits_snoop %lld \n", core,  cc_s.l3_hits_xsnp_hits[core]);
    }
  }
#endif
    free( l_tsc_timer );
  }

  /* free data */
  for ( i = 0; i < l_n_workers - 1; ++i )
    for ( j = 0; j < l_n_levels; ++j ) {
      free( l_n_buffers[i][j] );
  }
  free( l_startTime_arr );
  free( l_endTime_arr );
  free( l_iterTimes );

  return 0;
}
