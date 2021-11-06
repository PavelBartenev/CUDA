#include <iostream>
#include <CosineVector.cuh>
#include <CommonKernels.cuh>
#include <ScalarMul.cuh>
#include <KernelMul.cuh>
#include "ScalarMulRunner.cu"

float CosineVector(int numElements, float* vector1, float* vector2) {
  float* vec1_dev = nullptr;
  float* vec2_dev = nullptr;
  float* vec1_squared_dev = nullptr;
  float* vec2_squared_dev = nullptr;
  int size = numElements * sizeof(float);

  cudaMalloc(&vec1_dev, size);
  cudaMalloc(&vec2_dev, size);
  cudaMalloc(&vec1_squared_dev, size);
  cudaMalloc(&vec2_squared_dev, size);

  cudaMemcpy(vec1_dev, vector1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(vec2_dev, vector2, size, cudaMemcpyHostToDevice);

  float scalar_mul = ScalarMulSumPlusReduction(numElements, vec1_dev, vec2_dev, 512);

  int blockSize = 512;
  int num_blocks = (numElements + blockSize - 1) / blockSize;

  KernelMul <<<num_blocks, blockSize>>> (numElements, vec1_dev, vec1_dev, vec1_squared_dev);
  KernelMul <<<num_blocks, blockSize>>> (numElements, vec2_dev, vec2_dev, vec2_squared_dev);

  int cur_res_size = numElements;

  while (cur_res_size > 1) {
      int cur_grid_size = (cur_res_size + blockSize - 1) / blockSize;
      SumBlock <<<cur_grid_size, blockSize, blockSize*sizeof(float)>>>(cur_res_size, vec1_squared_dev, vec1_squared_dev);
      SumBlock <<<cur_grid_size, blockSize, blockSize*sizeof(float)>>>(cur_res_size, vec2_squared_dev, vec2_squared_dev);
      cur_res_size = cur_grid_size;
  }

  float vec1_squared_sum = 0;
  float vec2_squared_sum = 0;

  cudaMemcpy(&vec1_squared_sum, vec1_squared_dev, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&vec2_squared_sum, vec2_squared_dev, sizeof(float), cudaMemcpyDeviceToHost);

  return (scalar_mul / (sqrt(vec1_squared_sum) * sqrt(vec2_squared_sum)));
}

