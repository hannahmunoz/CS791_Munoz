#ifndef MULTIPLY_H_
#define MULTIPLY_H_

__global__ void kDistance (float* parsedCSV, int row, float* results);
__device__ int getGlobalIdx();
__device__ void bubbleSort(float results[], int iter);

#endif // MULTIPLY_H_