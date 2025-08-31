/*******************************************************************
 * Author: Neeraj , Shivam , Iqbaal
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	32	// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------


 /*  //loop unroll+reorder
   
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            double val = A[i * size + k];
            int j = 0;
            // Unroll the j loop by 8
            if(size>8)
            {
            for (; j <= size - 8; j += 8) {
                C[i * size + j]     += val * B[k * size + j];
                C[i * size + j + 1] += val * B[k * size + j + 1];
                C[i * size + j + 2] += val * B[k * size + j + 2];
                C[i * size + j + 3] += val * B[k * size + j + 3];
                C[i * size + j + 4] += val * B[k * size + j + 4];
                C[i * size + j + 5] += val * B[k * size + j + 5];
                C[i * size + j + 6] += val * B[k * size + j + 6];
                C[i * size + j + 7] += val * B[k * size + j + 7];
             }
            }
            // Handle remaining elements if size was not the multiple of 8
            for (; j < size; j++) {
                C[i * size + j] += val * B[k * size + j];
            }
        }
    }
 
 */



 //loop reorder kij
/*

 for (int k = 0; k < size; k++) {
    for (int i = 0; i < size; i++) {
        double val = A[i * size + k];
        for (int j = 0; j < size; j++) {
            C[i * size + j] += val * B[k * size + j];
        }
    }
}*/

 //Loop reorder   ikj 
        for (int i = 0; i < size; i++) 
        {
        for (int k = 0; k < size; k++) 
        {
            double val = A[i * size + k];
            for (int j=0; j < size; j++) {
                C[i * size + j] += val * B[k * size + j];
            }
        }
    }
    
     /*
*/    
   
	/*
	
	//loop unroll
    
        for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            int k = 0;

            // Unroll loop by 8
            for (; k <= size - 8; k += 8) {
                sum += A[i * size + k] * B[k * size + j];
                sum += A[i * size + k + 1] * B[(k + 1) * size + j];
                sum += A[i * size + k + 2] * B[(k + 2) * size + j];
                sum += A[i * size + k + 3] * B[(k + 3) * size + j];
                sum += A[i * size + k + 4] * B[(k + 4) * size + j];
                sum += A[i * size + k + 5] * B[(k + 5) * size + j];
                sum += A[i * size + k + 6] * B[(k + 6) * size + j];
                sum += A[i * size + k + 7] * B[(k + 7) * size + j];
            }

            // Handle remaining elements
            for (; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }

            C[i * size + j] += sum;
        }
    }*/
}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
/*
   for(int i=0 ; i<size ; i+=tile_size){
		for(int j=0 ; j<size ; j+= tile_size){
            for(int k=0 ; k<size ; k+= tile_size){
                for ( int i1 = i; i1 < i +  tile_size; i1++) {
			        for ( int j1 = j; j1 <j + tile_size; j1++) {
			        	for (int k1 = k; k1 <k + tile_size; k1++) {
			        		C[i1 * size + j1] += A[i1 * size + k1] * B[k1 * size + j1];
			        	}
			        }
		        }
            }
        }
    }  
	*/


    double *temp = (double *)malloc(size * size * sizeof(double));

    // Transpose B here (no separate function)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            temp[j * size + i] = B[i * size + j];
        }
    }

    // Tiled multiplication using temp for better locality
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int k = 0; k < size; k += tile_size) {
                for (int i1 = i; i1 < i + tile_size && i1 < size; i1++) {
                    for (int j1 = j; j1 < j + tile_size && j1 < size; j1++) {
                        double sum = C[i1 * size + j1];
                        for (int k1 = k; k1 < k + tile_size && k1 < size; k1++) {
                            sum += A[i1 * size + k1] * temp[j1 * size + k1];
                        }
                        C[i1 * size + j1] = sum;
                    }
                }
            }
        }
    }

    free(temp);


}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *a, double *b, double *c, int size) { //naive SIMD implementation using avx2 intrinsics
     
     //128bit transpose version
     
     double *temp = (double*) aligned_alloc(32, size * size * sizeof(double));// Allocate space for transposed B
	 for (int i = 0; i < size; i++) {      // Transpose B into temp
        for (int j = 0; j < size; j++) {
            temp[j * size + i] = b[i * size + j];
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m128d SUM = _mm_setzero_pd(); 					 // store the sum whatever calculated till and it initialized to zero
            int k;
            for (k = 0; k <= size - 2; k += 2) {
                __m128d A = _mm_loadu_pd(&a[i * size + k]);   // 2 doubles from row A
                __m128d B = _mm_loadu_pd(&temp[j * size + k]);  // 2 doubles from row of B transpose temp
				SUM = _mm_fmadd_pd(A, B, SUM);                 // accumulate (A*B + SUM) here we using fused multiplication and addition
            }                                                      // original SUM is [s0, s1]
            __m128d t1 = _mm_hadd_pd(SUM, SUM);                 // horizontal addition [s0+s1,s0+s1]only pairwise
            double s = _mm_cvtsd_f64(t1);                       // take only one of double from it
            for (; k < size; k++) {         					 // Handle remainder if matrix was not mutiple of 4
                s += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = s;
        }
    }free(temp);
    
    
    
   /*  
    // 256bit transpose version
     
   double *temp = (double*) aligned_alloc(32, size * size * sizeof(double));// Allocate space for transposed B
     for (int i = 0; i < size; i++) {      // Transpose B into temp
        for (int j = 0; j < size; j++) {
            temp[j * size + i] = b[i * size + j];
        }
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m256d SUM = _mm256_setzero_pd();                      // store the sum whatever calculated till and it initialized to zero
            int k;
            for (k = 0; k <= size - 4; k += 4) {
                __m256d A = _mm256_loadu_pd(&a[i * size + k]);   // 4 doubles from row A[i][k..k+3] simd loading
                __m256d B = _mm256_loadu_pd(&temp[j * size + k]);  // 4 doubles from row of B transpose temp[i][k..k+3] simd loading
                SUM = _mm256_fmadd_pd(A, B, SUM);                 // accumulate (A*B + SUM) here we using fused multiplication and addition
            }                                                      // original SUM is [s0, s1, s2, s3]
            __m256d t1 = _mm256_hadd_pd(SUM, SUM);                 // horizontal addition [s0+s1, s2+s3, s0+s1, s2+s3] only pairwise
            __m256d t2 = _mm256_permute2f128_pd(t1, t1, 1);        // swap 128-bit halves [s2+s3, s0+s1, s2+s3, s0+s1]
            __m256d t3 = _mm256_add_pd(t1, t2);                    // final sum in all of 4 part [all 4 contains the sum]
            double s = _mm256_cvtsd_f64(t3);                       // take only one of double from it
            for (; k < size; k++) {                              // Handle remainder if matrix was not mutiple of 4
                s += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = s;
        }
    }free(temp);

    */
    
    /*for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m256d SUM = _mm256_setzero_pd();  // store the sum whatever calculated till and it initialized to zero
            int k;
            for (k = 0; k <= size - 4; k += 4) {
                __m256d A = _mm256_loadu_pd(&a[i * size + k]);   // 4 doubles from row A[i][k..k+3] simd loading
                __m256d B = _mm256_set_pd(b[(k+3) * size + j], b[(k+2) * size + j], b[(k+1) * size + j], b[(k+0) * size + j]);// using set instead of load for filling array B directely (AVX2 required)
                //__m256d B = _mm256_i64gather_pd(&b[k * size + j], _mm256_set_epi64x(3*size, 2*size, 1*size, 0), 8);  this one another way to load. else do transpose of B and load using loadu
				// or use double temp[4]; and load each one to it and directely use loadu
				SUM = _mm256_fmadd_pd(A, B, SUM);                 // accumulate (A*B + SUM) here we using fused multiplication and addition
            }                                                      // original SUM is [s0, s1, s2, s3]
            __m256d t1 = _mm256_hadd_pd(SUM, SUM);                 // horizontal addition [s0+s1, s2+s3, s0+s1, s2+s3] only pairwise
            __m256d t2 = _mm256_permute2f128_pd(t1, t1, 1);        // swap 128-bit halves [s2+s3, s0+s1, s2+s3, s0+s1]
            __m256d t3 = _mm256_add_pd(t1, t2);                    // final sum in all of 4 part [all 4 contains the sum]
            double s = _mm256_cvtsd_f64(t3);                       // take only one of double from it
            for (; k < size; k++) {          // Handle remainder if matrix was not mutiple of 4
                s += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = s;
        }
    }*/
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
 
 //simd+tiling
 double *temp = (double*) aligned_alloc(32, size * size * sizeof(double));// Allocate space for transposed B
    for (int i = 0; i < size; i++) {// Transpose B into temp
        for (int j = 0; j < size; j++) {
            temp[i * size + j] = B[j * size + i];
        }
    }
    
    for (int M = 0; M < size; M += tile_size) {    // Perform matrix multiplication using tiling
        int Mlim = (M + tile_size < size) ? (M + tile_size) : size;

        for (int N = 0; N < size; N += tile_size) {
            int Nlim = (N + tile_size < size) ? (N + tile_size) : size;

            for (int O = 0; O < size; O += tile_size) {
                int Olim = (O + tile_size < size) ? (O + tile_size) : size;

                // Matrix tile multiplication
                for (int i = M; i < Mlim; i++) {
                    for (int j = N; j < Nlim; j++) {
                        __m256d SUM = _mm256_setzero_pd();  // Initialize sum to zero
                        int k;
                        for (k = O; k <= Olim - 4; k += 4) {
                            __m256d x = _mm256_loadu_pd(&A[i * size + k]);  // Load 4 elements from row A[i]
                            __m256d y = _mm256_loadu_pd(&temp[j * size + k]);  // Load 4 elements from row temp[j]
                            SUM = _mm256_fmadd_pd(x, y, SUM);         // Multiply and accumulate using FMA
                        }
                        // Horizontal addition of the SIMD result
                        __m256d t1 = _mm256_hadd_pd(SUM, SUM);  // Pairwise sum [s0+s1, s2+s3, s0+s1, s2+s3]
                        __m256d t2 = _mm256_permute2f128_pd(t1, t1, 1);  // Swap halves
                        __m256d t3 = _mm256_add_pd(t1, t2);  // Final sum
                        double s = _mm256_cvtsd_f64(t3);  // Extract the final sum

                         for (; k < Olim; k++) {  // Handle the remaining elements in the k loop ie when not multiple of 4 since 256 bit simd could only load 4 double
                            s += A[i * size + k] * temp[j * size + k];
                        }

                        C[i * size + j] += s; // Store the result in matrix C or result matrix
                    }
                }
            }
        }
    }
    free(temp); // Free the allocated memory for the transposed B matrix


/*
//tiling+reodering+simd
    for (int M = 0; M < size; M += tile_size) {
        int Mlim = (M + tile_size < size) ? (M + tile_size) : size;

        for (int O = 0; O < size; O += tile_size) {
            int Olim = (O + tile_size < size) ? (O + tile_size) : size;

            for (int N = 0; N < size; N += tile_size) {
                int Nlim = (N + tile_size < size) ? (N + tile_size) : size;

                for (int i = M; i < Mlim; i++) {
                    for (int k = O; k < Olim; k++) {
                        __m256d a = _mm256_set1_pd(A[i * size + k]);  // broadcast A[i,k]

                        int j = N;
                        for (; j + 4 <= Nlim; j += 4) {
                            __m256d b = _mm256_loadu_pd(&B[k * size + j]);  // B[k,j...j+3]
                            __m256d c = _mm256_loadu_pd(&C[i * size + j]);  // C[i,j...j+3]
                            c = _mm256_fmadd_pd(a, b, c);  // fused multiply-add
                            _mm256_storeu_pd(&C[i * size + j], c); //i guess ooptimization possible here. possibility of any idea of just keeping this in register as we calculating completely ?
                        }

                        for (; j < Nlim; j++) {
                            C[i * size + j] += A[i * size + k] * B[k * size + j];
                        }
                    }
                }
            }
        }
    }
    
    */
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		 //perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		 start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
