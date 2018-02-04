#include <iostream>
#include <stdlib.h> 
#include <cmath>
#include <stdio.h>

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
	float *MatA = new float[(int)pow(matDim, 2)];
	float *MatB = new float[(int)pow(matDim, 2)]; 
	float *MatC = new float[(int)pow(matDim, 2)];
	
	//load
	for (int i=0; i < (int)pow(matDim, 2); i++) {
 		MatA[i] = (float) i;
 		MatB[i] = (float) i;
 	}


	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

   	for (int i = 0; i < matDim; i++) {
      		for (int j = 0; j < matDim; j++) {
			float sum = 0.0;
        		for (int k = 0; k < matDim; k++) {
         			 sum = sum + MatA[i*matDim + k]*MatB[k*matDim+j];
       			}
 
       		 	MatC [i*matDim + j] = sum;
     	 	}

  	}

	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );


	//output results
	/*for (int i = 0; i < matDim; i++){
		for (int j = 0; j < matDim; j++){
			printf ("%.2f	", MatC[(i*matDim)+j]); 
			//std::cout << MatC[(i*matDim)+j] << " ";
		}
	std::cout << std::endl;
	}*/

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