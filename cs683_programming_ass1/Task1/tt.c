#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

void combination(double *A, double *B, double *C, int size, int tile_size) {
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
}


//print function
void print(double *m, int size) 
{
    return;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%8.2f ", m[i * size + j]);
        }
        printf("\n");
    }
}
// Naive O(n^3) matrix multiplication
void naive(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

int main(){
    clock_t start, end;
    int size = 1024;    //static array of this size initialization part  
    int tile_size = 64;  //tile size
    double *A = (double *)malloc(size * size * sizeof(double));
    double *B = (double *)malloc(size * size * sizeof(double));
    double *C1 = (double *)calloc(size * size, sizeof(double));
    double *C2 = (double *)calloc(size * size, sizeof(double));

     for (int i = 1; i <= size * size; i++) {
        A[i] = i;
        B[i] = i*2;
    }

    printf("\n\nMatrix A\n");
    print(A,size);
    printf("\n\nMatrix B\n");
    print(B,size);
    printf("\n\n\n");

    start = clock();
    naive(A, B, C1, size);
    end = clock();
    printf("Naive:     implementation    %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    print(C1,size);
    printf("\n\n");


     start = clock();
    combination(A, B, C2, size,tile_size);
    end = clock();
    printf("Naive:         %.3f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    print(C2,size);
    printf("\n\n");

    free(A); free(B); free(C1); free(C2);
    return 0;
}