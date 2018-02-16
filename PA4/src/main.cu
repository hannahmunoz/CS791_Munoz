#include <iostream>
#include <stdio.h>

#include <cmath>

#include "mat.h"

bool isSquare(int num){	return (floor (sqrt(num)) == sqrt(num));}
bool check (int argc, char* argv[]);
void fillMat (float *mat, int size);
void printMat (float *mat, int matDim,  int offset);

int main (int argc, char* argv[]){
	
	//variables
	int matDim, matnum, blockDim, threadDim, size;

	if (check (argc, argv)){
		return 1;
	}

	matDim = atoi (argv [1]);
	matnum = atoi(argv [2]);
	blockDim = atoi(argv [3]);
	threadDim = atoi(argv [4]);
	size = matDim * matDim * (matnum/2);

	// initalize more varaibles
	dim3 grid (blockDim, blockDim);
	dim3 block (threadDim, threadDim);

	//create arrays
	float *MatA = new float [size];
	float *MatB = new float [size]; 
	float *MatC = new float [matDim * matDim];

	cudaMallocManaged( (void**)&MatA, size * sizeof(float) );
	cudaMallocManaged( (void**)&MatB, size * sizeof(float) );
	cudaMallocManaged( (void**)&MatC, matDim * matDim * sizeof(float) );

	fillMat (MatA, size);
	fillMat (MatB, size);

	/*for (int i = 0; i < matnum/2; i ++){
		printMat (MatA, matDim, i*matDim*matDim);
	}*/



	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

	//add
	for (int i = 0; i < matnum/2; i ++){
		if (i % 2 ==0){
			add <<<grid, block>>> (MatA, MatB, MatC, i*matDim*matDim);
		}
		else{
			multiply <<<grid, block>>> (MatA, MatB, MatC, matDim, i*matDim*matDim);
		} 
	}
	

	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );
	
	printMat (MatC, matDim, 0);


 	float elapsedTime;
  	cudaEventElapsedTime( &elapsedTime, start, end );

        std::cout << "Time: " << elapsedTime << " ms." << std::endl;

	//dealloc memory
    	cudaEventDestroy( start );
        cudaEventDestroy( end );
	cudaFree (MatA);
	cudaFree (MatB);
	cudaFree (MatC);
}

bool check (int argc,char* argv[]){

	cudaDeviceProp prop;
 	cudaGetDeviceProperties( &prop, 0 );

	if (argc < 5){
		std::cout << "Not enough arguments. <<matrix dimension>> <<number of matrices>> << block dimension>> << thread dimension>>" << std::endl; 
		return true;
	}
	if (atoi (argv [1]) <=0 || atoi (argv [1]) >= 32000){
		std::cout << "Matrix dimension not valid. Must be between 0 and 32000." << std::endl;
		return true;
	}
	if (atoi (argv [2]) % 2 != 0){
		std::cout << "Even number of matrices needed" << std::endl;
		return true;
	}
	if ( atoi(argv [3]) <=0 || atoi(argv [3]) >= 25000 ){
		std::cout << "Block dimension not valid. Must be between 0 and 25000." << std::endl;
		return true;
	}
	if ( atoi(argv [4]) <=0 || atoi(argv [4]) > sqrt(prop.maxThreadsPerBlock) ){
		std::cout << "Thread dimension not valid. Must be between 0 and " << sqrt(prop.maxThreadsPerBlock)  << "." << std::endl;
		return true;
	}
	if ( atoi(argv [3])  * atoi(argv [4]) != atoi(argv [1])){
		std::cout << "Not enough/too many blocks and threads for given matrix dimensions" << std::endl;
		return true;
	}

	return false;
}

void fillMat (float *mat, int size){
	for (int i=0; i < size; i++) {
		mat [i] = (float)(rand()) /(RAND_MAX/100);
 	}
}

void printMat (float *mat, int matDim, int offset){

	for (int i = 0; i < matDim; i++){
		for (int j = 0; j < matDim; j++){
			printf ("%.02f	", mat[(i*matDim)+j + offset]); 
		}
		printf ("\n");
	}
	printf ("\n");

}
