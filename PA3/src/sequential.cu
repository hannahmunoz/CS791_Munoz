#include <iostream>
#include <stdlib.h> 
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream> 
#include <vector>
#include <time.h>
#include <algorithm>
#include <utility>

const int GLOBAL_CONST_ROW = 161;
const int GLOBAL_CONST_COL = 128;

void fileIn (std::string name, std::vector<float> &parsedCSV);

float kDistance(std::vector<float> parsedCSV, int row);

void euclidean (std::vector<float> parsedCSV, std::vector<float> &distance, int row);

int main (int argc, char* argv[]){
	//variables
	if (argc < 2){
		std::cout << "No file name given" << std::endl;
		return 1;
	}
	
	srand(1);

	// read in file

	std::vector<float> parsedCSV;

	fileIn (argv[1], parsedCSV);

	std::vector<std::pair <int, float> > results;

	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord(start, 0);

	for (int i = 0; i < GLOBAL_CONST_ROW; i++){
		if (isnan(parsedCSV[i*(GLOBAL_CONST_COL)])){

			results.push_back ( std::make_pair(i, kDistance (parsedCSV, i)));
		}
	
	}


	//end time
	cudaEventRecord(end, 0);
  	cudaEventSynchronize( end );


	for (int i = 0; i < results.size(); i++){
		printf ("Row, %d,	k, %.02f\n", results[i].first, results[i].second);

	}


	/*for (int i = 0; i < GLOBAL_CONST_ROW; i++){
		for (int j = 0; j < GLOBAL_CONST_COL; j++){
			std::cout << parsedCSV[i*(GLOBAL_CONST_COL)+j] << " ";
		}
		std::cout << std::endl << std::endl;
	}*/

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
			if ( iter % 128 == 1 && rand() % 10 == 1){
				parsedCSV.push_back(NAN);

			}
			else{
				parsedCSV.push_back(atof (s.c_str()));
			}
			iter++;
		}
	}
	file.close();

}

float kDistance(std::vector<float> parsedCSV, int row){

	std::vector<float> distance;

	euclidean (parsedCSV, distance, row);

	//sort the vector
        std::stable_sort (distance.begin(), distance.end());

	/*for (int i = 0; i < distance.size(); i++){
		std::cout << distance [i] << " ";
	}
	std::cout << std::endl;*/

	// k = 5
	distance.resize (5);
	float sum = 0.0;
	for (int i = 0; i < distance.size(); i++){
		sum += distance [i];
	}
	
	//std::cout << row << ": " << sum/5 << " ";
	
	//std::cout << std::endl << std::endl;

	return sum/5;
}

void euclidean (std::vector<float> parsedCSV, std::vector<float> &distance, int row){
	for (int i = 0; i < GLOBAL_CONST_ROW; i++){
		if ( i != row){
			float runningSum = 0.0;
			for (int j = 1; j < GLOBAL_CONST_COL; j++){
				float temp = parsedCSV[row*(GLOBAL_CONST_COL)+j] - parsedCSV[i*(GLOBAL_CONST_COL)+j];
				runningSum += temp * temp;
			}

			distance.push_back (sqrt(runningSum));
		}
	}

}