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
	int numGPU, matDim, matnum, blockDim, threadDim, size;

	if (check (argc, argv)){
		return 1;
	}
	numGPU = atoi (argv [1]);
	matDim = atoi (argv [2]);
	matnum = atoi(argv [3]);
	blockDim = atoi(argv [4]);
	threadDim = atoi(argv [5]);
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
	int i = 0;
	int j = 0;

	//add
	//for (int i = 0; i < matnum/2; i += numGPU){
		//for (int j = 0; j < numGPU; j +=2){

	#pragma omp parallel for
	while (i < matnum/2 && j < numGPU){
		cudaSetDevice (j);
		if (i % 2 == 0){
			add <<<grid, block>>> (MatA, MatB, MatC, i*matDim*matDim);
		}
		else{
			multiply <<<grid, block>>> (MatA, MatB, MatC, matDim, i*matDim*matDim);
		}		
	 	i++;
		j++;
		if ( j == numGPU){
			j = 0;
		}
	}

	/*for (int i = 0; i < matnum/2; i ++){
		if (i % 2 == 0){
			add <<<grid, block>>> (MatA, MatB, MatC, i*matDim*matDim);
		}
		else{
			multiply <<<grid, block>>> (MatA, MatB, MatC, matDim, i*matDim*matDim);
		}
	}*/
	

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
	int numGPU; 
	cudaGetDeviceCount(&numGPU);

	if (argc < 6){
		std::cout << "Not enough arguments. <<number of GPU>> <<matrix dimension>> <<number of matrices>> << block dimension>> << thread dimension>>" << std::endl; 
		return true;
	}
	if (atoi (argv [1]) <=0 || atoi (argv [1]) >= numGPU){
		std::cout << "Must have between 1 and " << numGPU << " GPUs" << std::endl;
		return true;
	}

	if (atoi (argv [2]) <=0 || atoi (argv [2]) >= 32000){
		std::cout << "Matrix dimension not valid. Must be between 0 and 32000." << std::endl;
		return true;
	}
	if (atoi (argv [3]) % 2 != 0){
		std::cout << "Even number of matrices needed" << std::endl;
		return true;
	}
	if ( atoi(argv [4]) <=0 || atoi(argv [4]) >= 25000 ){
		std::cout << "Block dimension not valid. Must be between 0 and 25000." << std::endl;
		return true;
	}
	if ( atoi(argv [5]) <=0 || atoi(argv [5]) > sqrt(prop.maxThreadsPerBlock) ){
		std::cout << "Thread dimension not valid. Must be between 0 and " << sqrt(prop.maxThreadsPerBlock)  << "." << std::endl;
		return true;
	}
	if ( atoi(argv [4])  * atoi(argv [5]) != atoi(argv [2])){
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
