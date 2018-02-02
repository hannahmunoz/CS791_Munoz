#ifndef ADD_H_
#define ADD_H_

__global__ void add(float*, float*, float*);
__device__ int getGlobalIdx();

#endif // ADD_H_