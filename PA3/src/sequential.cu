#include <iostream>
#include <stdlib.h> 
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream> 
#include <vector>
#include <time.h>


const int GLOBAL_CONST_ROW = 161;
const int GLOBAL_CONST_COL = 128;

void fileIn (std::string name, std::vector<float> &parsedCSV);

int main (int argc, char* argv[]){
	//variables
	if (argc < 2){
		std::cout << "No file name given" << std::endl;
		return 1;
	}
	
	srand(time(NULL));

	// read in file

	std::vector<float> parsedCSV;

	fileIn (argv[1], parsedCSV);


	for (int i = 0; i < GLOBAL_CONST_ROW; i++){
		for (int j = 0; j < GLOBAL_CONST_COL; j++){
			std::cout << parsedCSV[i*(GLOBAL_CONST_COL)+j] << " ";
		}
		std::cout << std::endl << std::endl;
	}


	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );
	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );


	

	// read out file

	//output results
	float elapsedTime;
  	cudaEventElapsedTime( &elapsedTime, start, end );

        std::cout << "Time: " << elapsedTime << " ms." << std::endl;


	return 0;

}

void fileIn (std::string name, std::vector<float> &parsedCSV){
	
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
			if ( iter % 129 == 1){
				if ( rand() % 10 == 1){
					parsedCSV.push_back(0.00);

				}
				else{
					parsedCSV.push_back(atof (s.c_str()));
				}
			}
			else{
				parsedCSV.push_back(atof (s.c_str()));
			}
		}
	}
	file.close();

}
