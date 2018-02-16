#include <stdio.h>

#include "mat.h"

__global__ void add (float *a, float *b, float *c, int offset){
	int globalPos = getGlobalIdx();
	c[globalPos] += a[globalPos+offset] + b[globalPos+offset];
}

__global__ void multiply (float *a, float *b, float *c, int size, int offset){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	while (row < size) {
		while (col < size){
			float temp = 0;
        		for (int i = 0; i < size; i++) {
            			temp += a[row * size + i+offset] * b[i * size + col+offset];
        		}
    			c[row * size + col] += temp;
			col+= blockDim.y * gridDim.y;
		}
		col = blockIdx.x*blockDim.x+threadIdx.x;
		row += blockDim.x * gridDim.x;
	} 
}


// from the striding paper
__device__ int getGlobalIdx(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}