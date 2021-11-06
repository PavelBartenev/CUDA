#include "KernelMatrixAdd.cuh"

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int col_index = blockIdx.x * blockDim.x + threadIdx.x;

  float* row_A = (float*)((char*)A + row_index * pitch);
  float* row_B = (float*)((char*)B + row_index * pitch);
  float* row_result = (float*)((char*)result + row_index * pitch);

  row_result[col_index] = row_A[col_index] + row_B[row_index];
}

