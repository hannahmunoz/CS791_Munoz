#include <stdio.h>

#include "multiply.h"

__global__ void multiply (float *a, float *b, float *c, int size){
	int globalPos = getGlobalIdx();
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;

	//printf ("global: %d \n (%d, %d)\n\n", globalPos, row, col);
	float temp = 0;
	if (row < size && col < size) {    
        	for (int i = 0; i < size; i++) {
            		temp += a[row * size + i] * b[i * size + col];
        	}
    	c[row * size + col] = temp;
	} 
}


// from the striding paper
__device__ int getGlobalIdx(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
