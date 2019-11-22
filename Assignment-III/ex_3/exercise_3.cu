#include <cuda.h>
#include <stdio.h>
#include <curand.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <cublas_v2.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#define THRESHOLD 1e-3

/* CUDA layout */
dim3 grid(1);
dim3 block(TILE_SIZE, TILE_SIZE);

/* from cuda samples */
void checkGpuError(cudaError_t result, char const *const func, const char *const file, int const line) {
        if(result!=cudaSuccess) { \
                fprintf(stderr, "Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(result));
                exit(1);
        }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) checkGpuError((val), #val, __FILE__, __LINE__)

/* https://gist.github.com/Tener/803377 */
#define CURAND_CALL(x) { \
	do { \
		if((x) != CURAND_STATUS_SUCCESS) { \
			printf("Error at %s:%d\n",__FILE__,__LINE__);            \
			exit(1); \
		} \
	} while(0); \
}

/* time diff in ms */
double elapsed(struct timeval t0, struct timeval t1)
{
	return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

/* compare matrix with abs difference */
void compare_matrix(float *matrix_a, float *matrix_b, long size, double threshold)
{
	for (long i = 0; i < size*size; i++) {
		if (fabs((double)matrix_a[i] - (double)matrix_b[i]) > threshold) {
			fprintf(stderr, "Compare matrix failed: %f vs %f\n", matrix_a[i], matrix_b[i]);
			exit(1);
		}
	}
}

/* init matrix with curand */
void init_matrix(float *matrix, long size, unsigned long long seed)
{
	float *d_matrix = NULL;
	curandGenerator_t gen;

	checkCudaErrors(cudaMalloc(&d_matrix, sizeof(float)*size*size));

	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
	CURAND_CALL(curandGenerateUniform(gen, d_matrix, size*size));

	checkCudaErrors(cudaMemcpy(matrix, d_matrix, sizeof(float)*size*size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_matrix));
	CURAND_CALL(curandDestroyGenerator(gen));
}

/* C = AB on CPU with re-ordered loop */
void cpu_sgemm(float *C, float *A, float *B, long size)
{
	struct timeval t0, t1;

	gettimeofday(&t0, NULL);

	for (long i = 0; i < size; i++) {
		for (long k = 0; k < size; k++) {
			for (long j = 0; j < size; j++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}

	gettimeofday(&t1, NULL);

	printf("CPU matmul:\t\t\t%f ms\n", elapsed(t0, t1));
}

/* matmul kernel with global memory */
__global__
void naive_sgemm_kernel(float *C, float *A, float *B, long size)
{
	const long i = blockIdx.x * blockDim.x + threadIdx.x;
	const long j = blockIdx.y * blockDim.y + threadIdx.y;
	float val = 0.0;

	if (i >= size || j >= size)
		return;

	for (long k = 0; k < size; k++) {
		val += A[i * size + k] * B[k * size + j];
	}

	C[i * size + j] += val;
}

/* matmul with global memory */
void naive_sgemm(float *C, float *A, float *B, long size)
{
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	naive_sgemm_kernel<<<grid, block>>>(C, A, B, size);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&t1, NULL);

	printf("GPU matmul (global memory):\t%f ms\n", elapsed(t0, t1));
}

/* matmul kernel with shared memory */
__global__
void shared_sgemm_kernel(float *C, float *A, float *B, long size)
{
	const long col = blockIdx.x * blockDim.x + threadIdx.x;
	const long row = blockIdx.y * blockDim.y + threadIdx.y;
	float val = 0.0;

	/* TODO declare shared memory with size TILE_SIZE x TILE_SIZE */
	__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_B[TILE_SIZE][TILE_SIZE];

	if (col < size && row < size) {
		const long local_col = blockIdx.x * TILE_SIZE + threadIdx.x;
		const long local_row = blockIdx.y * TILE_SIZE + threadIdx.y;

		for (long m = 0; m < size / TILE_SIZE; ++m) {
			tile_A[threadIdx.y][threadIdx.x] = A[local_row * size + (m * TILE_SIZE + threadIdx.x)];
			tile_B[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * size + local_col];
			__syncthreads();
	
			/* TODO introduce a pragma directive that can potentially improve performance here */
			#pragma unroll
			for (long k = 0; k < TILE_SIZE; ++k) {
				/* TODO Perform multiplication here */
				val += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
			}
			__syncthreads();
		}

		C[local_row * size + local_col] = val;
	}
}

/* matmul with shared memory */
void shared_sgemm(float *C, float *A, float *B, long size)
{
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	shared_sgemm_kernel<<<grid, block>>>(C, A, B, size);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&t1, NULL);

	printf("GPU matmul (shared memory):\t%f ms\n", elapsed(t0, t1));
}

/* cuBLAS */
void cublas_sgemm(float *C, float *A, float *B, long size)
{
	struct timeval t0, t1;
	float alpha = 1.0;
	float beta = 0.0;

	cublasHandle_t handle;
	cublasCreate(&handle);

	gettimeofday(&t0, NULL);
	/* TODO fill in the blanks, do C = BA instead of C = AB */
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, B, size, A, size, &beta, C, size);
	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&t1, NULL);
	cublasDestroy(handle);

	printf("GPU cuBLAS matmul:\t\t%f ms\n", elapsed(t0, t1));
}

void print_usage(char *program)
{
	fprintf(stderr, "Usage: %s [-s size] [-v to verify with CPU sgemm]\n", program);
}

int main(int argc, char *argv[])
{
	int opt;
	long size = 64;
	bool verify = false;

	while ((opt = getopt(argc, argv, "s:v")) != -1) {
		switch (opt) {
			case 's':
				size = atol(optarg);
				if (size % TILE_SIZE != 0) {
					fprintf(stderr, "Error: Matrix size must be a multiple of tile size %d.\n", TILE_SIZE);
					exit(1);
				}
				break;
			case 'v':
				verify = true;
				printf("Matrix size: %ldx%ld\n", size, size);
				break;
			default:
				print_usage(argv[0]);
				exit(1);
		}
	}

	grid = dim3(((size + (TILE_SIZE - 1)) / TILE_SIZE), ((size + (TILE_SIZE - 1)) / TILE_SIZE));

	printf("Matrix size: %ldx%ld\n", size, size);
	printf("Grid size: %ux%u\n", grid.x, grid.y);
	printf("Tile size: %ux%u\n", TILE_SIZE, TILE_SIZE);
	printf("Run CPU sgemm: %d\n\n", verify);

	float *A = (float*)malloc(sizeof(float)*size*size);
	float *B = (float*)malloc(sizeof(float)*size*size);
	float *C_result = (float*)malloc(sizeof(float)*size*size);
	float *C_truth = (float*)malloc(sizeof(float)*size*size);

	float *d_A = NULL; 
	float *d_B = NULL; 
	float *d_C = NULL;

	if (A == NULL || B == NULL || C_truth == NULL || C_result == NULL) {
		fprintf(stderr, "Error: %s\n", strerror(errno));
		exit(1);
	}

	/* initialize A and B */
	init_matrix(A, size, 1);
	init_matrix(B, size, 5);
	memset(C_truth, 0, sizeof(float)*size*size);

	/* allocate A and B on GPU */
	checkCudaErrors(cudaMalloc(&d_A, sizeof(float)*size*size));
	checkCudaErrors(cudaMalloc(&d_B, sizeof(float)*size*size));
	checkCudaErrors(cudaMalloc(&d_C, sizeof(float)*size*size));

	/* copy A and B to GPU */
	checkCudaErrors(cudaMemcpy(d_A, A, sizeof(float)*size*size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeof(float)*size*size, cudaMemcpyHostToDevice));

	/* host gemm */
	if (verify) {
		cpu_sgemm(C_truth, A, B, size);
	}

	/* set C on GPU and run cublas */
	checkCudaErrors(cudaMemset(d_C, 0, sizeof(float)*size*size));
	cublas_sgemm(d_C, d_A, d_B, size);
	if (verify) {
		checkCudaErrors(cudaMemcpy(C_result, d_C, sizeof(float)*size*size, cudaMemcpyDeviceToHost));
		compare_matrix(C_result, C_truth, size, THRESHOLD);
	}
	else {
		checkCudaErrors(cudaMemcpy(C_truth, d_C, sizeof(float)*size*size, cudaMemcpyDeviceToHost));
	}

	/* run naive gpu gemm */
	checkCudaErrors(cudaMemset(d_C, 0, sizeof(float)*size*size));
	naive_sgemm(d_C, d_A, d_B, size);
	checkCudaErrors(cudaMemcpy(C_result, d_C, sizeof(float)*size*size, cudaMemcpyDeviceToHost));
	compare_matrix(C_result, C_truth, size, THRESHOLD);

	/* run shared */
	checkCudaErrors(cudaMemset(d_C, 0, sizeof(float)*size*size));
	shared_sgemm(d_C, d_A, d_B, size);
	checkCudaErrors(cudaMemcpy(C_result, d_C, sizeof(float)*size*size, cudaMemcpyDeviceToHost));
	compare_matrix(C_result, C_truth, size, THRESHOLD);

	/* free */
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
	free(A);
	free(B);
	free(C_truth);
	free(C_result);

	return 0;
}
