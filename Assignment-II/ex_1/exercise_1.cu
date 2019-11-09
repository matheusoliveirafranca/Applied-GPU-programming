#include <stdio.h>
#define N 1
#define TPB 256


__global__ void helloWorldKernel()
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	printf("Hello World! My threadId is  %2d\n", i);
}

int main()
{

	// Launch kernel to print “Hello World!” and the thread identifier.
 	helloWorldKernel<<<N, TPB>>>();

  	// Synchronize device
  	cudaDeviceSynchronize();
  
  	return 0;
}