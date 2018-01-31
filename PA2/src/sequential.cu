#include <iostream>
#include <stdlib.h> 
#include <cmath>


int main (int argc, char* argv[]){
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
	int *MatA = new int[(int)pow(matDim, 2)];
	int *MatB = new int[(int)pow(matDim, 2)]; 
	int *MatC = new int[(int)pow(matDim, 2)];
	
	//load
	for (int i=0; i < (int)pow(matDim, 2); i++) {
 		MatA[i] = i;
 		MatB[i] = i;
 	}


	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );




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