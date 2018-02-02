#include <stdio.h>

#include "multiply.h"

__global__ void multiply (float *a, float *b, float *c){
	int globalPos = getGlobalIdx();
	//c[globalPos] = a[globalPos]+ b[globalPos];
		//printf ("%d\n", c[globalPos+i]);
   printf("block coor (%d, %d)\nthead coor(%d, %d)\nglobal coor (%d)\n grid coor (%d, %d) \n\n", blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, globalPos, gridDim.x, gridDim.y);
}

__global__ void dotProduct (int threadID, int col){
	

}


// from the striding paper
__device__ int getGlobalIdx(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

//__device__ int getNextID(int block