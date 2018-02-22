#include <stdio.h>

#include "mat.h"

__global__ void add (float *a, float *b, float *c, int offset){
	int globalPos = getGlobalIdx();
	c[globalPos] += a[globalPos+offset] + b[globalPos+offset];
}

__global__ void multiply (float *a, float *b, float *c, int size, int offset){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

        for (int k = 0; k < size; k++) {
            	c[row * size + col] += a[(row * size) + k+offset] * b[(k * size) + col+offset];
        }

}


// from the striding paper
__device__ int getGlobalIdx(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}