#include <iostream>
#include <stdlib.h> 
#include <cmath>

void add (float*, float*, float*, float);

int main (float argc, char* argv[]){
	//variables
	int matDim;

	// get inputs
	if (argc < 2){
		std::cout << "Not enough arguments. <<matrix dimension>>" << std::endl; 
		return 1;
	}
	else{
	       matDim = atoi (argv [1]);
	}

	//create arrays
	float *MatA = new float[(int)pow(matDim, 2)];
	float *MatB = new float[(int)pow(matDim, 2)]; 
	float *MatC = new float[(int)pow(matDim, 2)];
	
	//load
	for (int i=0; i < (float)pow(matDim, 2); i++) {
 		MatA[i] = i;
 		MatB[i] = i;
 	}

	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

	//add
	add (MatA, MatB, MatC, (int)pow(matDim, 2));

	//output results
	/*for (int i = 0; i < matDim; i++){
		for (int j = 0; j < matDim; j++){
			std::cout << MatC[(i*matDim)+j] << " ";
		}
	std::cout << std::endl;
	}*/

	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );

 	float elapsedTime;
  	cudaEventElapsedTime( &elapsedTime, start, end );

        std::cout << "Time: " << elapsedTime << " ms." << std::endl;

	//dealloc memory
	delete MatA;
	MatA = NULL;
	delete MatB;
	MatB = NULL;
	delete MatC;
	MatC = NULL;

	return 0;
}

void add (float* a, float* b, float* c, float size){
	for (int i=0; i < size; i++) {
		c [i] = a[i] + b[i];
 	}
}