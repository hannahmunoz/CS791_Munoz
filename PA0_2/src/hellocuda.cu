
#include <iostream>

#include "add.h"

int main() {
  
  int *dev_a, *dev_b, *dev_c;

  cudaError_t err = cudaMalloc( (void**) &dev_a, N * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
  cudaMallocManaged( (void**) &dev_a, N * sizeof(int));
  cudaMallocManaged( (void**) &dev_b, N * sizeof(int));
  cudaMallocManaged( (void**) &dev_c, N * sizeof(int));

  for (int i = 0; i < N; ++i) {
    dev_a[i] = i;
    dev_b[i] = i;
  }

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord( start, 0 );

  add<<<N, 1>>>(dev_a, dev_b, dev_c);

  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  for (int i = 0; i < N; ++i) {
    if (dev_c[i] != dev_a[i] + dev_b[i]) {
      std::cerr << "Oh no! Something went wrong. You should check your cuda install and your GPU. :(" << std::endl;

      // clean up events - we should check for error codes here.
      cudaEventDestroy( start );
      cudaEventDestroy( end );

      // clean up device pointers - just like free in C. We don't have
      // to check error codes for this one.
      cudaFree(dev_a);
      cudaFree(dev_b);
      cudaFree(dev_c);
      exit(1);
    }
  }

  std::cout << "Yay! Your program's results are correct." << std::endl;
  std::cout << "Your program took: " << elapsedTime << " ms." << std::endl;
  
  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

}
