BIN = mat_mul
CC = g++

tiling: mat_mul
	$(CC) -O0 mat_mul.c -mavx2 -mfma -D OPTIMIZE_TILING -o $(BIN)/$@

loop: mat_mul
	$(CC) -O0 mat_mul.c -mavx2 -mfma -D OPTIMIZE_LOOP_OPT -o $(BIN)/$@

naive: mat_mul
	$(CC) -O0 mat_mul.c -mavx2 -mfma -D NAIVE -o $(BIN)/$@

simd: mat_mul
	$(CC) -O0 mat_mul.c -mavx2 -mfma -D OPTIMIZE_SIMD -o $(BIN)/$@

combination: mat_mul
	$(CC) -O0 mat_mul.c -mavx2 -mfma -D OPTIMIZE_COMBINED -o $(BIN)/$@

all: mat_mul
	$(CC) -O0 mat_mul.c -mavx2 -mfma -D OPTIMIZE_TILING -D OPTIMIZE_SIMD -D OPTIMIZE_LOOP_OPT -D OPTIMIZE_COMBINED -o $(BIN)/$@

clean:
	@rm -rf $(BIN)
	@rm -f out.txt

mat_mul:
	@mkdir -p $(BIN)

