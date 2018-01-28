#include <iostream>
#include <cmath>

#include "add.h"

bool isSquare(int num){	return (floor (sqrt(num)) == sqrt(num));}

int main (int argc, char* argv[]){
	
	//variables
	int matDim, blockDim, threadDim;

	// get inputs
	if (argc < 4){
		std::cout << "Not enough arguments" << std::endl; 
		return 1;
	}
	else{
	       matDim = atoi (argv [1]);
	       blockDim = atoi(argv [2]);
	       threadDim = atoi(argv [3]);
	}

	
	// bounds checking
	if ( matDim <=0){
		std::cout << "Matrix dimension not valid" << std::endl;
		return 1;
	}
	if ( blockDim <=0 ){
		std::cout << "Block dimension not valid" << std::endl;
		return 1;
	}
	if ( threadDim <=0 ){
		std::cout << "Matrix dimension not valid" << std::endl;
		return 1;
	}
	if ( blockDim * threadDim < matDim){
		std::cout << "Not enough blocks and threads for given matrix dimensions" << std::endl;
		return 1;
	}

	// initalize more varaibles
	dim3 grid (blockDim, blockDim);
	dim3 block (threadDim, threadDim);
	int addsPerThread = (int)pow(matDim, 2)/((int)pow(blockDim, 2)* (int)pow(threadDim, 2));
	//std::cout << addsPerThread << std::endl;

	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );


	//create arrays
	int *MatA = new int[(int)pow(blockDim, 2)* (int)pow(threadDim, 2)];
	int *MatB = new int[(int)pow(blockDim, 2)* (int)pow(threadDim, 2)]; 
	int *MatC = new int[(int)pow(blockDim, 2)* (int)pow(threadDim, 2)];

	for (int i=0; i < (int)pow(blockDim, 2)* (int)pow(threadDim, 2); i++) {
 		MatA[i] = i;
 		MatB[i] = i;
 	}

	//alloc memory
	int *a, *b, *c;
	cudaMalloc( (void**)&a, pow(blockDim, 2)* pow(threadDim, 2) * sizeof(int) );
	cudaMalloc( (void**)&b, pow(blockDim, 2)* pow(threadDim, 2) * sizeof(int) );
	cudaMalloc( (void**)&c, pow(blockDim, 2)* pow(threadDim, 2) * sizeof(int) );

	//send to GPU
	cudaMemcpy (a, MatA, pow(blockDim, 2)* pow(threadDim, 2) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (b, MatB, pow(blockDim, 2)* pow(threadDim, 2) * sizeof(int), cudaMemcpyHostToDevice);

	//add
	add <<<grid, block>>> (a, b, c, addsPerThread);

	// get result from GPU
	cudaMemcpy (MatC, c, pow(blockDim, 2)* pow(threadDim, 2) * sizeof(int), cudaMemcpyDeviceToHost );

	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );

	for (int i = 0; i < matDim; i++){
		for (int j = 0; j < matDim; j++){
			std::cout << MatC[(i*matDim)+j] << " ";
		}
		std::cout << std::endl;
	}

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
