#include <iostream>
#include <ScalarMulRunner.cuh>
#include <CommonKernels.cuh>
#include "ScalarMul.cu"
#include "KernelMul.cu"

float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
    int num_blocks = (numElements + blockSize - 1) / blockSize;
    int vector_size = numElements * sizeof(float);

    float* vector1_dev = nullptr;
    float* vector2_dev = nullptr;

    cudaMalloc(&vector1_dev, vector_size);
    cudaMalloc(&vector2_dev, vector_size);

    cudaMemcpy(vector1_dev, vector1, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vector2_dev, vector2, vector_size, cudaMemcpyHostToDevice);

    float* result = (float*) malloc(num_blocks * sizeof(float));
    float* ans = (float*) malloc(num_blocks * sizeof(float));

    float* result_dev = nullptr;
    cudaMalloc(&result_dev, num_blocks * sizeof(float));

    ScalarMulBlock <<<num_blocks, blockSize, blockSize * sizeof(float)>>>(numElements, vector1_dev, vector2_dev, result_dev);

    cudaMemcpy(result, result_dev, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    int cur_res_size = num_blocks;

    while (cur_res_size > 1) {
        int cur_grid_size = (cur_res_size + blockSize - 1) / blockSize;
        SumBlock <<<cur_grid_size, blockSize, blockSize*sizeof(float)>>>(cur_res_size, result_dev, result_dev);
        cur_res_size = cur_grid_size;
    }

    cudaMemcpy(result, result_dev, sizeof(float), cudaMemcpyDeviceToHost);

    return result[0];
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
    int vector_size = numElements * sizeof(float);

    float* mul_array_host = (float*) malloc(vector_size);
    float* mul_array_dev = nullptr;
    cudaMalloc(&mul_array_dev, vector_size);

    float* vector1_dev = nullptr;
    float* vector2_dev = nullptr;

    cudaMalloc(&vector1_dev, vector_size);
    cudaMalloc(&vector2_dev, vector_size);

    cudaMemcpy(vector1_dev, vector1, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vector2_dev, vector2, vector_size, cudaMemcpyHostToDevice);

    int num_blocks = (numElements + blockSize - 1) / blockSize;

    KernelMul<<<num_blocks, blockSize>>>(numElements, vector1_dev, vector2_dev, mul_array_dev);

    cudaMemcpy(mul_array_host, mul_array_dev, vector_size, cudaMemcpyDeviceToHost);

    int cur_res_size = numElements;

    while (cur_res_size > 1) {
        int cur_grid_size = (cur_res_size + blockSize - 1) / blockSize;
        SumBlock <<<cur_grid_size, blockSize, blockSize*sizeof(float)>>>(cur_res_size, mul_array_dev, mul_array_dev);
        cur_res_size = cur_grid_size;
    }

    cudaMemcpy(mul_array_host, mul_array_dev, sizeof(float), cudaMemcpyDeviceToHost);

    return mul_array_host[0];
}

