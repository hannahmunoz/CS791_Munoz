#ifndef MULTIPLY_H_
#define MULTIPLY_H_

__global__ void multiply (float*, float*, float*);
__global__ void dotProduct (int, int);
__device__ int getGlobalIdx();

#endif // MULTIPLY_H_