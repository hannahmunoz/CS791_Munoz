#include <iostream>
#include <stdlib.h> 
#include <cmath>
#include <vector>
#include <stdio.h>
 

void fillMat (std::vector <std::vector <std::vector <float> > > &mat, int matDim, int matNumber);
void printMat (std::vector <std::vector <float> >  mat);
void add (std::vector <std::vector <float> >  a, std::vector <std::vector <float> > b, std::vector <std::vector <float> > &c);
void mult (std::vector <std::vector <float> >  a, std::vector <std::vector <float> > b, std::vector <std::vector <float> > &c);


int main (int argc, char* argv[]){
	//variables
	int matDim, matNumber;

	// get inputs
	if (argc < 3){
		std::cout << "Not enough arguments. <<matrix dimension>> <<number of matrices>>" << std::endl; 
		return 1;
	}
	else{
	       matDim = atoi (argv [1]);
	       matNumber = atoi (argv [2]);

	}
	if (matNumber % 2 != 0){
		std::cout << "Even number of matrices needed" << std::endl;
		return 1;
	}


	srand(1);


	//create arrays
	std::vector <std::vector <std::vector <float> > > MatA;
	std::vector <std::vector <std::vector <float> > > MatB;
	std::vector <std::vector <float> > MatC;

	fillMat (MatA, matDim, matNumber/2);
	fillMat (MatB, matDim, matNumber/2);

	for (int i = 0; i < matDim; i++){
		std::vector <float> temp (matDim, 0);
		MatC.push_back (temp);
	}


	// begin timing
 	cudaEvent_t start, end;
  	cudaEventCreate(&start);
  	cudaEventCreate(&end);

 	cudaEventRecord( start, 0 );

	for (int i = 0; i < matNumber/2; i++){
		if (i % 2 == 0){
			add (MatA [i], MatB [i], MatC);
		}
		else{
			mult (MatA [i], MatB [i], MatC);
		}
	}


	//end time
	cudaEventRecord( end, 0 );
  	cudaEventSynchronize( end );

	//print results
	printMat (MatC);

 	float elapsedTime;
  	cudaEventElapsedTime( &elapsedTime, start, end );

        std::cout << "Time: " << elapsedTime << " ms." << std::endl;

	return 0;
}

void fillMat (std::vector <std::vector <std::vector <float> > >  &mat, int matDim, int matNumber){
	for (int i=0; i < matNumber; i++) {
		std::vector <std::vector <float> > singleMat;	
		for (int j = 0; j < matDim; j++){
			std::vector <float> temp;
			for (int k = 0; k < matDim; k++){
				temp.push_back ( (float)(rand()) /(RAND_MAX/100));
			}
			singleMat.push_back (temp);
		}
		mat.push_back (singleMat);
 	}

}

void printMat (std::vector <std::vector <float> > mat){
	
	for (int i=0; i < mat.size(); i++) {
		for (int j = 0; j < mat[i].size(); j++){
			printf ("%.02f %*c", mat[i][j], 5, ' ');
		}
		printf ("\n");
 	}
	printf ("\n");
}


void add (std::vector <std::vector <float> >  a, std::vector <std::vector <float> > b, std::vector <std::vector <float> > &c){
	for (int i=0; i < a.size(); i++) {
		for (int j = 0; j < a[i].size(); j++){
			c [i][j] += a[i][j] + b[i][j];
		}
 	}
}

void mult (std::vector <std::vector <float> >  a, std::vector <std::vector <float> > b, std::vector <std::vector <float> > &c){
   	for (int i = 0; i < a.size(); i++) {
      		for (int j = 0; j < a.size(); j++) {
			float sum = 0.0;
        		for (int k = 0; k < a.size(); k++) {
         			 sum += a[i][k] * b[k][j];
       			}
 
       		 	c[i][j] += sum;
     	 	}
  	}

}