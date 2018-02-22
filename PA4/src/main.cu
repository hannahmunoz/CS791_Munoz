#include <iostream>
#include <stdio.h>
#include <cmath>

#include "mat.h"
#include "book.h"


struct DataStruct
{
	int deviceID;
	int blocks; 
	int threads;
	int matDim;
	int offset;
	float *MatA; 
	float *MatB; 
	float *MatC;
	float returnValue; 
};

bool isSquare(int num){	return (floor (sqrt(num)) == sqrt(num));}
bool check (int argc, char* argv[]);
void fillMat (float *mat, int size);
void printMat (float *mat, int matDim,  int offset);
void* addroutine (void *tData);
void* multroutine (void *tData);

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
	//dim3 grid (blockDim, blockDim);
	//dim3 block (threadDim, threadDim);

	float *MatA;
	float *MatB;
	float *MatC;


	cudaMallocManaged( (void**)&MatA, size * sizeof(float) );
	cudaMallocManaged( (void**)&MatB, size * sizeof(float) );
	cudaMallocManaged( (void**)&MatC, matDim * matDim * sizeof(float) );

	fillMat (MatA, size);
	fillMat (MatB, size);


	DataStruct* threadData= new DataStruct[matnum/2];
	CUTThread * thread = new CUTThread[matnum/2];


	//for (int i = 0; i < numGPU; i+=2){
	for (int i = 0, j = 0; i < matnum/2 && j < numGPU; i++,j++){
		threadData[i].deviceID = j;
		threadData[i].blocks = blockDim; 
		threadData[i].threads = threadDim; 
		threadData[i].matDim = matDim;
		threadData[i].offset = i* matDim *matDim;
		threadData[i].MatA = MatA;
		threadData[i].MatB = MatB;
		threadData[i].MatC = MatC;
		if (j+1 == numGPU){
			j = 0;
		}
	}


	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

	
	for (int i = 0, j = 0; i < matnum/2 && j < numGPU; i++){

		if ( i%2 == 0){
			thread[i] = start_thread(addroutine, &threadData[i]);
		}
		else{
			thread[i] = start_thread(multroutine, &threadData[i]);
		}

		j++;
		if (j == numGPU){
			j = 0;
		}

	}
	
	wait_for_threads (thread,  matnum/2);


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
	/*if (atoi (argv [1]) <=1 || atoi (argv [1]) >= numGPU){
		std::cout << "Must have between 2 and " << numGPU << " GPUs" << std::endl;
		return true;
	}*/

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
	printf ("\n");
	for (int i = 0; i < matDim; i++){
		for (int j = 0; j < matDim; j++){
			printf ("%.02f	", mat[(i*matDim)+j + offset]); 
		}
		printf ("\n");
	}
	printf ("\n");

}

void* addroutine (void *tData){
	DataStruct *data = (DataStruct*)tData;
	cudaSetDevice(data->deviceID);

	//create arrays
	dim3 grid (data->blocks, data->blocks);
	dim3 block (data->threads, data->threads);
	
	float *C =  (float*)malloc(  data->matDim*data->matDim*sizeof(float) );

	float *partC;
	cudaMalloc( (void**)&partC, data->matDim*data->matDim*sizeof(float) );
	cudaMemcpy( partC, C, data->matDim*data->matDim*sizeof(float), cudaMemcpyHostToDevice );


	add <<<grid, block>>> (data->MatA, data->MatB, partC, data->offset);

	cudaMemcpy( C, partC, data->matDim*data->matDim*sizeof(float), cudaMemcpyDeviceToHost );

	for (int i=0; i< data->matDim*data->matDim; i++) {
        	data->MatC[i] += C[i];
    	}

	cudaFree( partC);
	free (C);
 
	
	return 0;

}

void* multroutine (void *tData){
	DataStruct *data = (DataStruct*)tData;

	cudaSetDevice(data->deviceID);

	dim3 grid (data->blocks, data->blocks);
	dim3 block (data->threads, data->threads);

	
	float *C =  (float*)malloc(  data->matDim*data->matDim*sizeof(float) );

	float *partC;
	cudaMalloc( (void**)&partC, data->matDim*data->matDim*sizeof(float) );
	cudaMemcpy( partC, C, data->matDim*data->matDim*sizeof(float), cudaMemcpyHostToDevice );

	multiply <<<grid, block>>> (data->MatA, data->MatB, partC, data->matDim, data->offset);

	cudaMemcpy( C, partC, data->matDim*data->matDim*sizeof(float), cudaMemcpyDeviceToHost );

	for (int i=0; i< data->matDim*data->matDim; i++) {
        	data->MatC[i] += C[i];
    	}

	cudaFree( partC);
	free (C);
 

	return 0;

}

