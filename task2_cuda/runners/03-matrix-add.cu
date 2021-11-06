#include <iostream>
#include <cstdio>
#include <cmath>
#include <KernelMatrixAdd.cuh>
#include <vector>


int main() {
  int width = 128;
  int height = 128;

  float* A_host = (float*)malloc(width * height * sizeof(float));
  float* B_host = (float*)malloc(width * height * sizeof(float));
  float* C_host = (float*)malloc(width * height * sizeof(float));

  for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
          A_host[row * width + col] = 1.f;
          B_host[row * width + col] = 3.f;
          C_host[row * width + col] = 0.f;
      }
  }

  float* A_dev = nullptr;
  float* B_dev = nullptr;
  float* C_dev = nullptr;
  size_t pitch_A_dev = 0;
  size_t pitch_B_dev = 0;
  size_t pitch_C_dev = 0;

  cudaMallocPitch(&A_dev, &pitch_A_dev, width * sizeof(float), height);
  cudaMallocPitch(&B_dev, &pitch_B_dev, width * sizeof(float), height);
  cudaMallocPitch(&C_dev, &pitch_C_dev, width * sizeof(float), height);

  cudaMemcpy2D(A_dev, pitch_A_dev, A_host, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(B_dev, pitch_B_dev, B_host, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(C_dev, pitch_C_dev, C_host, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

  std::vector <float> times;

  for (int matrix_size = 1; matrix_size < 1024; ++matrix_size) {

      int width = matrix_size;
      int height = matrix_size;

      int block_size = 16;

      float* A_dev = nullptr;
      float* B_dev = nullptr;
      float* C_dev = nullptr;
      size_t pitch_A_dev = 0;
      size_t pitch_B_dev = 0;
      size_t pitch_C_dev = 0;

      cudaMallocPitch(&A_dev, &pitch_A_dev, width * sizeof(float), height);
      cudaMallocPitch(&B_dev, &pitch_B_dev, width * sizeof(float), height);
      cudaMallocPitch(&C_dev, &pitch_C_dev, width * sizeof(float), height);

      dim3 dimBlock(block_size, block_size);
      dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);

      float time = 0;
      cudaEvent_t start, stop;

      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      KernelMatrixAdd <<<dimGrid, dimBlock>>>(height, width, pitch_A_dev, A_dev, B_dev, C_dev);

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&time, start, stop);

      times.push_back(time);
  }

  for (auto& time : times) {
      std::cout << time << " ";
  }

  cudaMemcpy2D(C_host, width * sizeof(float), C_dev, pitch_C_dev, width * sizeof(float), height, cudaMemcpyDeviceToHost);

  int error = 0;

  for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
          if (fabs(C_host[row * width + col] - 4.f) > 0.1f) {
              ++error;
          }
      }
  }

  std::cout << error;

  return 0;
}
