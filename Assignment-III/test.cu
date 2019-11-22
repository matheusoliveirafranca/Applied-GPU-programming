#include <stdio.h>
#define N 1
#define TPB 256
#define BDIMX 20
#define BDIMY 25
#define BLOCK_SIZE  16

__global__ void setRowReadColDyn(int *out)
{
    // dynamic shared memory
    extern __shared__ int tile[];
    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx]; 
}


int main()
{
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change

    int *d_C = { 0 };
    int image_size = BDIMX * BDIMY;

    cudaMalloc(&d_C, image_size * sizeof(int));
    cudaMemset(d_C, 0, image_size * sizeof(int));


    // Launch kernel to print “Hello World!” and the thread identifier.
    setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);

    // Synchronize device
    cudaDeviceSynchronize();

    for (int i = 0; i < BDIMX ; i++){
        for(int j = 0 ; j < BDIMY ; j++){
            printf("%d ", d_C[i*BDIMX + j]);
        }
        printf("\n");
    }
  
    return 0;
}