#ifndef MAT_H_
#define MAT_H_

__global__ void add(float*, float*, float*, int);
__global__ void multiply (float *a, float *b, float *c, int size, int offset);
__device__ int getGlobalIdx();

#endif // MAT_H_