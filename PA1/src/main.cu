#include <iostream>
#include <stdio.h>

#include <cmath>

#include "add.h"

bool isSquare(int num){	return (floor (sqrt(num)) == sqrt(num));}

int main (int argc, char* argv[]){
	
	//variables
	int matDim, blockDim, threadDim;

	// get inputs
	if (argc < 4){
		std::cout << "Not enough arguments. <<matrix dimension>> << block dimension>> << thread dimension>>" << std::endl; 
		return 1;
	}
	else{
	       matDim = atoi (argv [1]);
	       blockDim = atoi(argv [2]);
	       threadDim = atoi(argv [3]);
	}

	cudaDeviceProp prop;
 	cudaGetDeviceProperties( &prop, 0 );

	// bounds checking
	if ( matDim <=0 || matDim >= 32000){
		std::cout << "Matrix dimension not valid. Must be between 0 and 32000." << std::endl;
		return 1;
	}
	if ( blockDim <=0 || blockDim >= 25000 ){
		std::cout << "Block dimension not valid. Must be between 0 and 25000." << std::endl;
		return 1;
	}
	if ( threadDim <=0 || threadDim > sqrt(prop.maxThreadsPerBlock) ){
		std::cout << "Thread dimension not valid. Must be between 0 and " << sqrt(prop.maxThreadsPerBlock)  << "." << std::endl;
		return 1;
	}
	if ( blockDim * threadDim != matDim){
		std::cout << "Not enough/too many blocks and threads for given matrix dimensions" << std::endl;
		return 1;
	}

	// initalize more varaibles
	dim3 grid (blockDim, blockDim);
	dim3 block (threadDim, threadDim);

	//create arrays
	int *MatA = new int[(int)pow(matDim, 2)];
	int *MatB = new int[(int)pow(matDim, 2)]; 
	int *MatC = new int[(int)pow(matDim, 2)];

	for (int i=0; i < (int)pow(matDim, 2); i++) {
 		MatA[i] = i;
 		MatB[i] = i;
 	}

	//alloc memory
	int *a, *b, *c;
	cudaMalloc( (void**)&a,(int)pow(matDim, 2) * sizeof(int) );
	cudaMalloc( (void**)&b, (int)pow(matDim, 2) * sizeof(int) );
	cudaMalloc( (void**)&c, (int)pow(matDim, 2) * sizeof(int) );

	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

	//send to GPU
	cudaMemcpy (a, MatA, (int)pow(matDim, 2) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (b, MatB, (int)pow(matDim, 2) * sizeof(int), cudaMemcpyHostToDevice);

	//add
	add <<<grid, block>>> (a, b, c);

	// get result from GPU
	cudaMemcpy (MatC, c, (int)pow(matDim, 2) * sizeof(int), cudaMemcpyDeviceToHost );

	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );

	//for testing output
	/*for (int i = 0; i < matDim; i++){
		for (int j = 0; j < matDim; j++){
			std::cout << MatC[(i*matDim)+j] << " ";
		}
		std::cout << std::endl;
	}*/

 	float elapsedTime;
  	cudaEventElapsedTime( &elapsedTime, start, end );

        std::cout << "Time: " << elapsedTime << " ms." << std::endl;


	//dealloc memory
    	cudaEventDestroy( start );
        cudaEventDestroy( end );
	cudaFree (a);
	cudaFree (b);
	cudaFree (c);
	delete MatA;
	MatA = NULL;
	delete MatB;
	MatB = NULL;
	delete MatC;
	MatC = NULL;
}
