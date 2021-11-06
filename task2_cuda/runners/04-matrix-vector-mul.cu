#include <iostream>
#include <cstdio>
#include <cmath>
#include <MatrixVectorMul.cuh>
#include <vector>

int main() {
    int height = 1000;
    int width = 1000;

    float* A   = (float*) malloc(height * width * sizeof(float));
    float* vec = (float*) malloc(width * sizeof(float));
    float* res = (float*) malloc(height * sizeof(float));

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            A[i * width + j] = 1.0;

    for (int i = 0; i < width; ++i) {
        vec[i] = 2;
        res[i] = 0;
    }

    float* A_dev = nullptr;
    float* vec_dev = nullptr;
    float* res_dev = nullptr;

    cudaMalloc(&A_dev, height * width * sizeof(float));
    cudaMalloc(&vec_dev, width * sizeof(float));
    cudaMalloc(&res_dev, height * sizeof(float));

    cudaMemcpy(A_dev, A, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_dev, vec, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res_dev, res, width * sizeof(float), cudaMemcpyHostToDevice);

    std::vector <float> times;

    for (int n = 1; n < 1024; ++n) {
        int height = n;
        int width = n;

        float* A_dev = nullptr;
        float* vec_dev = nullptr;
        float* res_dev = nullptr;

        cudaMalloc(&A_dev, height * width * sizeof(float));
        cudaMalloc(&vec_dev, width * sizeof(float));
        cudaMalloc(&res_dev, height * sizeof(float));

        float time = 0;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        int threadsPerBlock = 256;
        int blocksPerGrid = (width + threadsPerBlock - 1) / threadsPerBlock;

        MatrixVectorMul <<<threadsPerBlock, blocksPerGrid, threadsPerBlock * sizeof(float)>>>(height, width, A_dev,
                                                                                              vec_dev, res_dev);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        times.push_back(time);
    }

    for (auto& time : times) {
        std::cout << time << " ";
    }

    cudaMemcpy(res, res_dev, width * sizeof(float), cudaMemcpyDeviceToHost);

    int error = 0;

    for (int i = 0; i < height; ++i) {
        if (fabs(res[i] - 2*width) > 0.1)
            ++error;
    }

    free(A);
    free(vec);
    free(res);

    cudaFree(A_dev);
    cudaFree(vec_dev);
    cudaFree(res_dev);

    std::cout << error << " ";
    std::cout << time;

    return 0;
}

