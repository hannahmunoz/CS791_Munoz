#include <iostream>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <fstream> 
#include <string>
#include <utility>

#include "kthnearestneighbor.h"

const int GLOBAL_CONST_ROW = 161;
const int GLOBAL_CONST_COL = 128;

void fileIn (std::string name, float* parsedCSV);

int main (int argc, char* argv[]){

	cudaDeviceProp prop;
 	cudaGetDeviceProperties( &prop, 0 );

	//variables
	int blockDim, threadDim;

	// get inputs
	if (argc < 4){
		std::cout << "Not enough arguments. <<filename>> << block dimension>> << thread dimension>>" << std::endl; 
		return 1;
	}
	else{
	       blockDim = atoi(argv [2]);
	       threadDim = atoi(argv [3]);
	}
	if (blockDim*threadDim <  sqrt(GLOBAL_CONST_ROW)){
		std::cout << "error: blocks and threads must cover the input file" << std::endl;
		std::cout << "must equal " << (int)sqrt(GLOBAL_CONST_ROW) << std::endl;
		return 1;
	} 


	srand(1);

	// initalize more varaibles
	dim3 grid (blockDim, blockDim);

	dim3 block (threadDim , threadDim );

	//create vector
	float* parsedCSV;
	float* results;

	//alloc memory
	cudaMallocManaged( (void**)&parsedCSV, GLOBAL_CONST_ROW * GLOBAL_CONST_COL * sizeof(float) );
	cudaMallocManaged( (void**)&results, GLOBAL_CONST_ROW * sizeof(float) );

	fileIn (argv[1], parsedCSV);

	/*for (int i = 0; i < GLOBAL_CONST_ROW; i++){
		for (int j = 0; j < GLOBAL_CONST_COL; j++){
			printf ("%.02f ",parsedCSV[i*(GLOBAL_CONST_COL)+j] );
		}
		printf ("\n");
	}*/

	std::vector<std::pair <int, float> > kresults;


	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

	for (int i = 0; i < GLOBAL_CONST_ROW; i++){
		if (isnan(parsedCSV[i*(GLOBAL_CONST_COL)])){
			kDistance <<<grid, block>>> (parsedCSV, i, results);
			printf ("Row, %d,	k, %.02f\n", i, results[0]);
		}
	
	}

	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );

	for (int i = 0; i < kresults.size(); i++){
		printf ("Row, %d,	k, %.02f\n", kresults[i].first, kresults[i].second);

	}

	//for testing output

 	float elapsedTime;
  	cudaEventElapsedTime( &elapsedTime, start, end );

        std::cout << "Time: " << elapsedTime << " ms." << std::endl;


	//dealloc memory
    	cudaEventDestroy( start );
        cudaEventDestroy( end );
	cudaFree (parsedCSV);
	cudaFree (results);

}

void fileIn (std::string name, float* parsedCSV){
	
	std::ifstream file (name.c_str());
	std::string s;

	// discard metadata on top
	if (file.good()){
		for (int i = 0; i < 9; i++){
			getline (file, s);
		}
	
		getline(file, s, ',');
		int iter = 1;
		while (getline(file, s, ',')) {
			if ( iter % 128 == 1 && rand() % 10 == 1){
				parsedCSV[iter-1] = NAN;

			}
			else{
				parsedCSV[iter-1] = atof (s.c_str());
			}
			iter++;
		}
	}
	file.close();

}
