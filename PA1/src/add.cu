#include <stdio.h>

#include "add.h"

__global__ void add (int *a, int *b, int *c, int addsPerThread){
	int globalPos = getGlobalIdx();
	for (int i = 0; i < addsPerThread; i++){
		c[globalPos+i] = a[globalPos+i]+ b[globalPos+i];
		//printf ("%d\n", c[globalPos+i]);
	} 
	
	
   	// printf("block coor (%d, %d)\nthead coor(%d, %d)\n \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

// from the striding paper
__device__ int getGlobalIdx(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}