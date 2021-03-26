/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "profile.h"

typedef struct gemm_def {
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  double alpha;
  double beta;
  int trans_a;
  int trans_b;
  int aligned_a;
  int aligned_c;
  int prefetch;
  int br_type;
  libxsmm_blasint br_count;
  int br_unroll;
  int tc_config;
} gemm_def;

int g_reps = 0;

LIBXSMM_INLINE void print_help(void) {
  printf("\n\n");
  printf("1. Usage (dense*dense=dense, correctness and performance):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    NUM_A\n");
  printf("    NUM_B\n");
  printf("    NUM_ROW_TEAM\n");
  printf("    NUM_COLUMN_TEAM\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2_BL2viaC, curAL2_BL2viaC\n");
  printf("\n\n");
}

LIBXSMM_INLINE
double run_jit_bfloat16( const gemm_def*         i_gemm_def,
                         libxsmm_bfloat16** i_a,
                         libxsmm_bfloat16** i_b,
                         libxsmm_bfloat16** o_c,
                         size_t num_a,
                         size_t num_b,
                         const unsigned int      i_print_jit_info,
                         size_t num_threads,
                         size_t _rt,
                         size_t _ct  ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
    /* nothing */
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);
#if 0
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
#endif
  l_start = libxsmm_timer_tick();
  size_t row_teams = _rt;
  size_t column_teams = _ct;
  size_t m_blocks = num_a;
  size_t n_blocks = num_b;

  #pragma omp parallel
  {
    if (i_gemm_def->tc_config) {
      cfg_tr.bsmm(NULL, NULL, NULL);
    }
    size_t ltid = omp_get_thread_num();
    size_t l_chunkSize = num_b * num_threads / num_threads;

    size_t my_col_id = ltid % column_teams; // 0
    size_t my_row_id = ltid / column_teams; //0
    size_t im_tasks_per_thread = (m_blocks + row_teams-1)/row_teams;
    size_t in_tasks_per_thread = (n_blocks + column_teams-1)/column_teams;
    size_t my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, m_blocks);
    size_t my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, m_blocks);
    size_t my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, n_blocks);
    size_t my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, n_blocks);


    for (size_t l_t = 0; l_t < g_reps; l_t++) {
      if (ltid == 0) SimMarker(1, l_t);
      for (size_t i = my_im_start; i < my_im_end; i++){
	for (size_t j = my_in_start; j < my_in_end; j++){
	  l_test_jit.bmrs(i_a[i], i_b[j], o_c[i*n_blocks+j], &l_br);
	}
      }
      if (ltid == 0) SimMarker(2, l_t);
    }
    if (i_gemm_def->tc_config) {
      rls_tr.bsmm(NULL, NULL, NULL);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}


int main(int argc, char* argv []) {
  char* l_precision = "BF16";
  libxsmm_blasint l_lda = 32, l_ldb = 32, l_ldc = 32;
  int l_m = 0, l_n = 0, l_k = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_trans_a = 0;
  int l_trans_b = 0;
  double l_alpha = 1;
  double l_beta = 1;
  int l_br = 1;
  int l_br_type = 0;
  int l_br_unroll = 0;
  int l_num_a = 1;
  int l_num_b = 1;
  size_t l_rt = 1;
  size_t l_ct = 1;

  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_matdiff_info l_diff;
  gemm_def l_gemm_def;
  size_t l_t = 0;
  double l_runtime_c = 0;
  double l_runtime_libxsmm = 0;
  libxsmm_timer_tickint l_start;
  int l_run_check = 0;

  /* input data */
  double *l_a_d = 0, *l_b_d = 0, *l_c_d = 0;
  float *l_a_f = 0, *l_b_f = 0, *l_c_f = 0;
  short *l_a_w = 0, *l_b_w = 0;
  libxsmm_bfloat16 **l_a_bf = 0, **l_b_bf = 0, **l_c_bf = 0;
  unsigned char *l_ua_b = 0, *l_ub_b;
  char *l_sa_b = 0, *l_sb_b = 0;
  int* l_c_b_i = 0;
  int* l_c_w_i = 0;
  unsigned char* l_c_b_ub = 0;
  float* l_c_bf_f = 0;
  /* Gold data */
  double* l_c_gold_d = 0;
  float* l_c_gold_f = 0;
  libxsmm_bfloat16* l_c_gold_bf = 0;
  int* l_c_gold_w_i = 0;
  int* l_c_gold_b_i = 0;
  unsigned char* l_c_gold_b_ub = 0;
  float* l_c_gold_bf_f = 0;
  double l_total_max_error = 0.0;
  int l_file_input = 0;
  int l_tc_config = 0;

  /* scaling factor */
  float l_scf = 1.0;

  libxsmm_matdiff_clear(&l_diff);

  /* check argument count for a valid range */
  if ( argc == 15 ) {
    /* xgemm sizes */
    l_m = atoi(argv[1]);
    l_n = atoi(argv[2]);
    l_k = atoi(argv[3]);
    l_num_a = atoi(argv[4]);
    l_num_b = atoi(argv[5]);
    l_rt = atoi(argv[6]);
    l_ct = atoi(argv[7]);

    /* some sugar */
    l_aligned_a = atoi(argv[8]);
    l_aligned_c = atoi(argv[9]);

    /* arch specific stuff */
    l_br = atoi(argv[10]);
    l_br_unroll = atoi(argv[11]);
    g_reps = atoi(argv[12]);
    l_tc_config = atoi(argv[13]);

    /* set value of prefetch flag */
    if (strcmp("nopf", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    }
    else if (strcmp("pfsigonly", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_SIGONLY;
    }
    else if (strcmp("BL2viaC", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;
    }
    else if (strcmp("curAL2", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    }
    else if (strcmp("curAL2_BL2viaC", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
    }
    else if (strcmp("AL2", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2;
    }
    else if (strcmp("AL2_BL2viaC", argv[14]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    l_run_check = 0;
  }
  else
  {
    print_help();
    return EXIT_FAILURE;
  }

  const char *env_arch = getenv("LIBXSMM_TARGET");
  const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
  int arch_cpuid = libxsmm_cpuid();

  int l_num_threads = 1;
  #pragma omp parallel
  {
    #pragma omp master
    l_num_threads = omp_get_num_threads();

  }

  printf("num_threads %d \n", l_num_threads);

  if ((!is_env_SPR && arch_cpuid < LIBXSMM_X86_AVX512_SPR)
       && (l_tc_config)) {
    printf("Warning: external tile configuration will be ingnored\n");
    l_tc_config = 0;
  }

  /* check alpha */
  if ( LIBXSMM_NEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "JIT: alpha needs to be 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "JIT: beta needs to be 0.0 or 1.0!\n");
    exit(EXIT_FAILURE);
  }

  if ( l_num_a % l_rt != 0  ) {
    fprintf(stderr, " row team has to be the factor of num of A tile \n");
    exit(EXIT_FAILURE);
  }

  if ( l_num_b % l_ct != 0  ) {
    fprintf(stderr, " column team has to be the factor of num of B tile \n");
    exit(EXIT_FAILURE);
  }

  if ( l_trans_b == 0 ) {
    printf("------------------------------------------------\n");
    printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
    printf("------------------------------------------------\n");
  } else {
    printf("------------------------------------------------\n");
    printf("RUNNING (%ix%i) X (%ix%i)^T = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
    printf("------------------------------------------------\n");
  }

  const int l_k_block = 2;
  double l_max_error = 0;
  int l_k2;

  l_gemm_def.m = l_m;
  l_gemm_def.n = l_n;
  l_gemm_def.k = l_k;
  l_gemm_def.lda = l_lda;
  l_gemm_def.ldb = l_ldb;
  l_gemm_def.ldc = l_ldc;
  l_gemm_def.alpha = l_alpha;
  l_gemm_def.beta = l_beta;
  l_gemm_def.trans_a = l_trans_a;
  l_gemm_def.trans_b = l_trans_b;
  l_gemm_def.aligned_a = l_aligned_a;
  l_gemm_def.aligned_c = l_aligned_c;
  l_gemm_def.prefetch = l_prefetch;
  l_gemm_def.br_type = l_br_type;
  l_gemm_def.br_count = l_br;
  l_gemm_def.br_unroll = l_br_unroll;
  l_gemm_def.tc_config = l_tc_config;


  l_a_bf = (libxsmm_bfloat16**)libxsmm_aligned_malloc((size_t)l_num_a * sizeof(libxsmm_bfloat16*), 64);
  l_b_bf = (libxsmm_bfloat16**)libxsmm_aligned_malloc((size_t)l_num_b * sizeof(libxsmm_bfloat16*), 64);
  l_c_bf = (libxsmm_bfloat16**)libxsmm_aligned_malloc((size_t)l_num_a * (size_t)l_num_b * sizeof(libxsmm_bfloat16*), 64);

  for ( int i = 0; i < l_num_a; i++)
    l_a_bf[i] = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
  for ( int i = 0; i < l_num_b * l_num_threads; i++)
    l_b_bf[i] = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
  for ( int i = 0; i < l_num_a * l_num_b * l_num_threads; i++){
    l_c_bf[i] = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
  }
  /* touch A */
  for (int i = 0; i < l_num_a; i++) {
    for (int l_r = 0; l_r < l_br; l_r++) {
      for (int l_i = 0; l_i < l_lda; l_i++) {
        for (int l_j = 0; l_j < l_k; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          l_a_bf[i][(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
        }
      }
    }
  }
  /* touch B */
  for (int i = 0; i < l_num_b; i++) {
    for (int l_r = 0; l_r < l_br; l_r++) {
      for (int l_i = 0; l_i < l_ldb; l_i++) {
        for (int l_j = 0; l_j < l_n; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          l_b_bf[i][(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
        }
      }
    }
  }
   /* touch C */
  for (int i = 0; i < l_num_a * l_num_b; i++) {
    for (int l_i = 0; l_i < l_ldc; l_i++) {
      for (int l_j = 0; l_j < l_n; l_j++) {
        union libxsmm_bfloat16_hp tmp;
        tmp.f = 0.0f;
        l_c_bf[i][(l_j * l_ldc) + l_i] = tmp.i[1];
      }
    }
  }

  l_runtime_libxsmm = run_jit_bfloat16( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf, l_num_a, l_num_b, l_file_input, l_num_threads, l_rt, l_ct);

  printf("%fs for C\n", l_runtime_c);
  printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
  printf("%fs for libxsmm\n", l_runtime_libxsmm);
  printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
  printf("max. error: %f\n", l_max_error);

  if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
    l_total_max_error = l_max_error;
  }

  libxsmm_free(l_a_bf);
  libxsmm_free(l_b_bf);
  libxsmm_free(l_c_bf);

  printf("\n\n Total Max Error %f\n\n", l_total_max_error );

  if ( l_total_max_error >= 0.00005 && l_br_type == 0) {
    return EXIT_FAILURE;
  } else if ( l_total_max_error >= 0.0005 && l_br_type > 0) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

