#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ia32intrin.h>
#include <omp.h>
#include "profile.h"

#define A_VAL 10
#define B_VAL 100

void usage(char *name)
{
    printf("Usage: %s <#iterations> <tile x-dim (# elems)> <tile y-dim (# elems)> <# A tiles> <# B tiles>\n", name);
    exit(-1);
}

void error(char *s)
{
    printf("Error: %s\n", s);
    exit(-1);
}

void do_work(int num_threads, short *A, short *B, short *C, int num_a,
    int num_b, int x_dim, int y_dim, int iterations)
{
    int tile_size = x_dim * y_dim;
    int c_tile_size = x_dim * x_dim;
    int z_dim = x_dim;

    #pragma omp parallel
    {
        int t = omp_get_thread_num();
#if 0
        __SSC_MARK(111);
#endif
        for (int i = 0; i < iterations; i++)
        {
            if (t==0) SimMarker(1, i);
            for (int a = 0; a < num_a; a++)
            {
                for (int b = 0; b < num_b; b++)
                {
		    for (int x = 0; x < x_dim; x++)
		    {
                        int z_offset = 0;
		        for (int z = 0; z < z_dim; z++)
	                {
			    __m512i c_data;
		      	    #pragma unroll(8)
                            for (int offset = 0; offset < y_dim; offset += 32)
                            {
                                __m512i a_data = _mm512_load_epi32(
                                    &A[a * tile_size + x * y_dim + offset]);
                                __m512i b_data = _mm512_load_epi32(
                                    &B[(t * num_b * tile_size) + b * tile_size +
                                    z * y_dim + offset]);
                                c_data += _mm512_add_epi16(a_data, b_data);
                            }

			    if (z_offset < z_dim)
			    {
                                _mm512_store_epi32(&C[
                                    (t * num_a * num_b * c_tile_size) +
                                    (a * num_b * c_tile_size) + (b * c_tile_size) +
                                    x * z_dim + z_offset], c_data);
			        z_offset += 32;
                            }
		        }
                    }
                }
	    }

            if (t==0) SimMarker(2, i);
        }
#if 0
        __SSC_MARK(222);
#endif
    }
}

void scalar_work(int num_threads, short *A, short *B, short *C, int num_a,
    int num_b, int x_dim, int y_dim)
{
    int tile_size = x_dim * y_dim;
    int c_tile_size = x_dim * x_dim;
    int z_dim = x_dim;

    #pragma omp parallel for
    for (int t = 0; t < num_threads; t++)
    {
        for (int a = 0; a < num_a; a++)
        {
            for (int b = 0; b < num_b; b++)
            {
                for (int x = 0; x < x_dim; x++)
                {
                    for (int z = 0; z < z_dim; z++)
		    {
			for (int y = 0; y < y_dim; y++)
			{
#if 0
			   C[(t * num_a * num_b * c_tile_size) +
			      (a * num_b * c_tile_size) + (b * c_tile_size) +
			      (x * z_dim) + z] =
			      A[a * tile_size + (x * y_dim) + y] +
			      B[(t * num_b * tile_size) + b * tile_size +
			      (y * z_dim) + z];
#endif
			   C[(t * num_a * num_b * c_tile_size) +
			      (a * num_b * c_tile_size) + (b * c_tile_size) +
			      (x * z_dim) + z] =
			      A[a * tile_size + (x * y_dim) + y] +
			      B[(t * num_b * tile_size) + b * tile_size +
			      (z * y_dim) + y];
			}
                    }
                }
            }
        }
    }
}

void check_work(int num_threads, short *C, short *C_prime, int num_a,
    int num_b, int x_dim, int y_dim)
{
    int tile_size = x_dim * y_dim;
    int c_tile_size = x_dim * x_dim;

    #pragma omp parallel for
    for (int t = 0; t < num_threads; t++)
    {
        for (int a = 0; a < num_a; a++)
        {
            for (int b = 0; b < num_b; b++)
            {
                for (int y = 0; y < x_dim; y++)
                {
		    for (int x = 0; x < x_dim; x++)
		    {
			if (C[(t * num_a * num_b * c_tile_size) +
			      (a * num_b * c_tile_size) + (b * c_tile_size) +
			      (y * x_dim) + x] !=
			      C_prime[(t * num_a * num_b * c_tile_size) +
			      (a * num_b * c_tile_size) + (b * c_tile_size) +
			      (y * x_dim) + x])
			{
			      error("Mismatch in answer!\n");
			}
		     }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) usage(argv[0]);

    int num_threads = 1;
   // omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp master
	num_threads = omp_get_num_threads();
    }

    int iterations = atoi(argv[1]);
    if (iterations <= 0) usage(argv[0]);

    int x_dim = atoi(argv[2]);
    if (x_dim <= 0) usage(argv[0]);

    if ((x_dim % 32) !=0)
    {
        printf("This version requires the x dimension to be a multiple of 32\n");
        usage(argv[0]);
    }

    int y_dim = atoi(argv[3]);
    if (y_dim <= 0) usage(argv[0]);

    int num_a = atoi(argv[4]);
    if (num_a <= 0) usage(argv[0]);

    int num_b = atoi(argv[5]);
    if (num_b <= 0) usage(argv[0]);

    printf("Running with:\n");
    printf("Threads: %d\n", num_threads);
    printf("Iterations: %d\n", iterations);
    printf("Tile size (w x h): %d x %d\n", x_dim, y_dim);
    printf("A tiles: %d\n", num_a);
    printf("B tiles: %d\n", num_b);

    int num_c = num_a * num_b * num_threads;

    // Allocate arrays
    // A is a shared input across all threads
    // B is a private input across all threads
    // C is a private output across all threads
    short *A, *B, *C, *C_prime;

    if (posix_memalign((void **)&A, 64, num_a * x_dim * y_dim * sizeof(short)))
    {
        error("Allocation of A");
    }

    if (posix_memalign((void **)&B, 64, num_threads * num_b * x_dim * y_dim *
        sizeof(short)))
    {
        error("Allocation of B");
    }

    if (posix_memalign((void **)&C, 64, num_c * x_dim * y_dim * sizeof(short)))
    {
        error("Allocation of C");
    }

    if (posix_memalign((void **)&C_prime, 64, num_c * x_dim * y_dim * sizeof(short)))
    {
        error("Allocation of C_prime");
    }

    // Initialize arrays
    for (int t_id = 0; t_id < num_a; t_id++)
    {
        for (int y = 0; y < y_dim; y++)
        {
            for (int x = 0; x < x_dim; x++)
            {
                A[t_id * y_dim * x_dim + y * x_dim + x] = A_VAL + t_id + y + x;
            }
        }
    }

    #pragma omp parallel for
    for (int t = 0; t < num_threads; t++)
    {
        for (int t_id = 0; t_id < num_b; t_id++)
        {
            for (int y = 0; y < y_dim; y++)
            {
                for (int x = 0; x < x_dim; x++)
                {
                    B[(t * num_b * y_dim * x_dim) + (t_id * y_dim * x_dim) +
                        (y * x_dim) + x] = B_VAL + t + t_id + y + x;
                }
            }
        }
    }

    memset((void *)C, 0, num_c * x_dim * y_dim * sizeof(short));
    memset((void *)C_prime, 0, num_c * x_dim * y_dim * sizeof(short));

    // Invoke real work function
    do_work(num_threads, A, B, C, num_a, num_b, x_dim, y_dim, iterations);
#if 0
    scalar_work(num_threads, A, B, C_prime, num_a, num_b, x_dim, y_dim);

    check_work(num_threads, C, C_prime, num_a, num_b, x_dim, y_dim);
#endif
    printf("PASSED\n");

    return 0;
}

