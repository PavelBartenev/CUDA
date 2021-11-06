#include <iostream>
#include <cstdio>
#include <cmath>
#include "KernelAdd.cuh"
#include <vector>


void measure_time_threads() {
    int N = 1 << 20;
    int size = N * sizeof(float);

    float* device_A = nullptr;
    float* device_B = nullptr;
    float* device_C = nullptr;
    cudaMalloc(&device_A, size);
    cudaMalloc(&device_B, size);
    cudaMalloc(&device_C, size);

    std::vector <float> times;

    for (int threadsPerBlock = 1; threadsPerBlock < 512; ++threadsPerBlock) {
        float time = 0;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        KernelAdd <<<blocksPerGrid, threadsPerBlock>>>(N, device_A, device_B, device_C);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        times.push_back(time);
    }

    for (auto& time : times) {
        std::cout << time << " ";
    }
}

void measure_time_vec_size() {
    int N = 1 << 20;
    int size = N * sizeof(float);

    float* device_A = nullptr;
    float* device_B = nullptr;
    float* device_C = nullptr;
    cudaMalloc(&device_A, size);
    cudaMalloc(&device_B, size);
    cudaMalloc(&device_C, size);

    std::vector <float> times;

    int threadsPerBlock = 256;

    for (int n = 1; n < N; n += 100) {
        float time = 0;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        KernelAdd <<<blocksPerGrid, threadsPerBlock>>>(n, device_A, device_B, device_C);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        times.push_back(time);
    }

    for (auto& time : times) {
        std::cout << time << " ";
    }
}


int main() {
    int64_t N = 1 << 20;
    size_t size = N * sizeof(float);

    float* host_A = (float*)malloc(size);
    float* host_B = (float*)malloc(size);
    float* host_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        host_A[i] = 1;
        host_B[i] = 2;
    }

    float* device_A = nullptr;
    float* device_B = nullptr;
    float* device_C = nullptr;
    cudaMalloc(&device_A, size);
    cudaMalloc(&device_B, size);
    cudaMalloc(&device_C, size);

    cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    int error = 0;
    std::cout << N << "\n";

    for (int i = 0; i < N; ++i) {
        if (fabs(host_C[i] - 3) > 0.01){
            error+=1;
        }
    }

    std::cout << error;

    free(host_A);
    free(host_B);
    free(host_C);

    return 0;
}


