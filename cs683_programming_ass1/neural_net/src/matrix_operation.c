#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j< m ; j++) {
			for(int l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
   	for (size_t i = 0; i < n; ++i) {
		for (size_t p = 0; p < k; ++p) {
			double val = A(i, p); // Assuming operator()(i, j) is defined
			for (size_t j = 0; j < m; ++j) {
				C(i, j) += val * B(p, j);
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
    size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 16; // unroll factor
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            size_t p = 0;

            // Unrolled loop (by 24)
            for (; p + UNROLL - 1 < k; p += UNROLL) {
                sum += A(i, p)      * B(p, j);
                sum += A(i, p + 1)  * B(p + 1, j);
                sum += A(i, p + 2)  * B(p + 2, j);
                sum += A(i, p + 3)  * B(p + 3, j);
                sum += A(i, p + 4)  * B(p + 4, j);
                sum += A(i, p + 5)  * B(p + 5, j);
                sum += A(i, p + 6)  * B(p + 6, j);
                sum += A(i, p + 7)  * B(p + 7, j);
                sum += A(i, p + 8)  * B(p + 8, j);
                sum += A(i, p + 9)  * B(p + 9, j);
                sum += A(i, p + 10) * B(p + 10, j);
                sum += A(i, p + 11) * B(p + 11, j);
                sum += A(i, p + 12) * B(p + 12, j);
                sum += A(i, p + 13) * B(p + 13, j);
                sum += A(i, p + 14) * B(p + 14, j);
                sum += A(i, p + 15) * B(p + 15, j);
               /**sum += A(i, p + 16) * B(p + 16, j);
                sum += A(i, p + 17) * B(p + 17, j);
                sum += A(i, p + 18) * B(p + 18, j);
                sum += A(i, p + 19) * B(p + 19, j);
                sum += A(i, p + 20) * B(p + 20, j);
                sum += A(i, p + 21) * B(p + 21, j);
                sum += A(i, p + 22) * B(p + 22, j);
                sum += A(i, p + 23) * B(p + 23, j);*/
            }

            // Handle remaining elements
            for (; p < k; ++p) {
                sum += A(i, p) * B(p, j);
            }

            C(i, j) = sum;
        }
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}


// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 32;   // tile size
	//int i_max = 0;
	//int k_max = 0;
	//int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    for (size_t i = 0; i < n; i += T) {
        for (size_t j = 0; j < m; j += T) {
            for (size_t p = 0; p < k; p += T) {
                // Tile boundaries
                size_t i_max = std::min(i + T, n);
                size_t j_max = std::min(j + T, m);
                size_t p_max = std::min(p + T, k);

                for (size_t i1 = i; i1 < i_max; ++i1) {
                    for (size_t j1 = j; j1 < j_max; ++j1) {
                        double sum = C(i1, j1);  // Assuming operator() access
                        for (size_t p1 = p; p1 < p_max; ++p1) {
                            sum += A(i1, p1) * B(p1, j1);
                        }
                        C(i1, j1) = sum;
                    }
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            __m256d SUM = _mm256_setzero_pd();  // zero initialize vector accumulator
            size_t p = 0;

            // Process 4 elements per iteration
            for (; p + 3 < k; p += 4) {
                // Load 4 elements from row i of A
                __m256d vecA = _mm256_loadu_pd(&A(i, p)); 

                // Load 4 elements from column j of B
                // Since B is column-major access for SIMD, manually load elements in reverse order for _mm256_set_pd
                __m256d vecB = _mm256_set_pd(
                    B(p + 3, j), 
                    B(p + 2, j), 
                    B(p + 1, j), 
                    B(p, j)
                );

                // Fused multiply-add: SUM += vecA * vecB
                SUM = _mm256_fmadd_pd(vecA, vecB, SUM);
            }

            // Horizontal add to sum all 4 doubles in SUM
            __m256d hadd1 = _mm256_hadd_pd(SUM, SUM);
            __m256d perm = _mm256_permute2f128_pd(hadd1, hadd1, 1);
            __m256d total = _mm256_add_pd(hadd1, perm);
            double s = _mm256_cvtsd_f64(total);

            // Process remaining elements if k is not divisible by 4
            for (; p < k; ++p) {
                s += A(i, p) * B(p, j);
            }

            C(i, j) = s;
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(j, i) = A(i, j);
		}
	}

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    size_t block_size = 270;  // Block size for cache optimization
    for (size_t i = 0; i < rows; i += block_size) {
        for (size_t j = 0; j < cols; j += block_size) {
            size_t i_max = std::min(i + block_size, rows);
            size_t j_max = std::min(j + block_size, cols);
            for (size_t k = i; k < i_max; ++k) {
                for (size_t m = j; m < j_max; ++m) {
                    result(m, k) = A(k, m);  // Directly write the transposed value
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
           
	
	return result;
}
