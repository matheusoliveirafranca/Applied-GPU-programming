#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/time.h>

#define TPB 256
#define EPSILON 0.0005
#define NTIME 1
#define ARRAY_SIZE 163840000

unsigned long get_time();

unsigned long get_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        unsigned long ret = tv.tv_usec;
        ret /= 1000;
        ret += (tv.tv_sec * 1000);
        return ret;
}


__global__ void SAXPYgpuKernel(float *x, float *y, float a)
{
  	const long i = blockIdx.x*blockDim.x + threadIdx.x;
  	y[i] = x[i]*a + y[i];
}

void SAXPYcpu(float* x, float* y, float a){

	for (long i = 0; i < ARRAY_SIZE ; i++){
		y[i] = x[i]*a + y[i];
	}
}

bool equalVectors(float* a, float* b){

	for (long i = 0; i < ARRAY_SIZE; i++){
		if (std::abs(a[i] - b[i]) > EPSILON){
			return false;
		}
	}

	return true;
}

int main()
{
	// seed for random number
	srand (static_cast <unsigned> (time(0)));


	// declare constant a
	const float a = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

  	// Declare pointers for an array of floats
	float* x = (float*) malloc (sizeof(float)*ARRAY_SIZE);
	float* y = (float*) malloc (sizeof(float)*ARRAY_SIZE);

	// set random values
	for (long i = 0 ; i < ARRAY_SIZE ; ++i){
		x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}


	// CPU execution
	long start_time_cpu = get_time();
	float* x_cpu = (float*) malloc (sizeof(float)*ARRAY_SIZE);
	float* y_cpu = (float*) malloc (sizeof(float)*ARRAY_SIZE);

	// copy vector to use in the CPU
	std::memcpy(x_cpu, x, ARRAY_SIZE*sizeof(float));
	std::memcpy(y_cpu, y, ARRAY_SIZE*sizeof(float));


	printf("Computing SAXPY on the CPU…");
	SAXPYcpu(x_cpu, y_cpu, a);
	printf("Done\n");
	long end_time_cpu = get_time();
  
  	// GPU execution

  	// Declare pointers for an array of floats
  	long start_time_gpu = get_time();
	float* x_gpu = 0;
	float* y_gpu = 0;
	float* y_gpu_res = (float*) malloc (sizeof(float)*ARRAY_SIZE);
	

	// Allocate device memory
	cudaMalloc(&x_gpu, ARRAY_SIZE*sizeof(float));
	cudaMalloc(&y_gpu, ARRAY_SIZE*sizeof(float));

	// Copy array to device
	cudaMemcpy(x_gpu, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	// Launch kernel to compute SAXPY
	printf("Computing SAXPY on the GPU...");
	SAXPYgpuKernel<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(x_gpu, y_gpu, a);
	
	// Synchronize device
	cudaDeviceSynchronize();

	// Copy back from device to CPU
	cudaMemcpy(y_gpu_res, y_gpu, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Done\n");

	long end_time_gpu = get_time();

	// Compare results
	printf("Comparing the output for each implementation for ARRAY_SIZE = %d; Comparison: ", ARRAY_SIZE);
	equalVectors(y_gpu_res, y_cpu) ? printf("Correct\n") : printf("Uncorrect\n");

	printf("CPU time: %ld ms\n", end_time_cpu-start_time_cpu);
	printf("GPU time: %ld ms\n\n", end_time_gpu-start_time_gpu);

	// Free the memory
	cudaFree(x_gpu);
	cudaFree(y_gpu);

	free(x);
	free(y);
	free(x_cpu);
	free(y_cpu);
	free(y_gpu_res);
	
	return 0;
}