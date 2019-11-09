#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#define ARRAY_SIZE 1000
#define TPB 256
#define EPSILON 0.005

__global__ void SAXPYgpuKernel(float *x, float *y, float a)
{
  	const int i = blockIdx.x*blockDim.x + threadIdx.x;
  	y[i] = x[i]*a + y[i];
}

void SAXPYcpu(float* x, float* y, float a){

	for (int i = 0; i < ARRAY_SIZE ; i++){
		y[i] = x[i]*a + y[i];
	}
}

bool equalVectors(float* a, float* b){

	for (int i = 0; i < ARRAY_SIZE; i++){
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
	for (int i = 0 ; i < ARRAY_SIZE ; ++i){
		x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		y[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}


	// CPU execution
	float* x_cpu = (float*) malloc (sizeof(float)*ARRAY_SIZE);
	float* y_cpu = (float*) malloc (sizeof(float)*ARRAY_SIZE);

	// copy vector to use in the CPU
	std::memcpy(x_cpu, x, ARRAY_SIZE*sizeof(float));
	std::memcpy(y_cpu, y, ARRAY_SIZE*sizeof(float));


	printf("Computing SAXPY on the CPU…");
	SAXPYcpu(x_cpu, y_cpu, a);
	printf("Done\n");
  
  	// GPU execution

  	// Declare pointers for an array of floats
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
	printf("Computing SAXPY on the GPU");
	SAXPYgpuKernel<<<(ARRAY_SIZE+TPB-1)/TPB, TPB>>>(x_gpu, y_gpu, a);
	
	// Synchronize device
	cudaDeviceSynchronize();

	// Copy back from device to CPU
	cudaMemcpy(y_gpu_res, y_gpu, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Done\n");


	// Compare results
	printf("Comparing the output for each implementation…");
	equalVectors(y_gpu_res, y_cpu) ? printf("Correct\n") : printf("Uncorrect\n");


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