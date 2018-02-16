#ifndef MAT_H_
#define MAT_H_

__global__ void add(float*, float*, float*);
__device__ int getGlobalIdx();

#endif // MAT_H_