#include <iostream>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <fstream> 
#include <string>

#include "kthnearestneighbor.h"

const int GLOBAL_CONST_ROW = 161;
const int GLOBAL_CONST_COL = 128;

__global__ void kDistance (float* parsedCSV, int row, float* results, float* kresults ){
 	//float results [GLOBAL_CONST_COL];
	int idx = getGlobalIdx();
	if (idx < GLOBAL_CONST_ROW){
	// get euclidean distance
	if (idx != row){
		int runningSum = 0.0;
		for (int j = 0; j < GLOBAL_CONST_COL; j++){
			float temp = parsedCSV[row*(GLOBAL_CONST_COL)+j] - parsedCSV[idx *(GLOBAL_CONST_COL)+j];
			runningSum += temp * temp;	
		}
		results [idx] = sqrtf (runningSum);


	}

	//sort

	if (idx == row){
		bubbleSort(results, GLOBAL_CONST_ROW);

		for (int i = 0; i < 5; i++){
			kresults[row] += results[i];
		}
		
		kresults[row] /= 5;
 	}
	}
}


// from the striding paper
__device__ int getGlobalIdx(){
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ void bubbleSort(float results[], int iter){
   	if (iter == 1)
      	  return;
 
    	for (int i = 0; i < iter-1; i++){
        	if (results[i] > results [i+1]){
			float temp = results[i+1];
			results[i+1] = results[i];
			results[i] = temp;
		}
 	}

       bubbleSort(results, iter-1);
}

 



