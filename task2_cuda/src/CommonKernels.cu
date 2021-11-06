#include <CommonKernels.cuh>

__global__
void SumBlock(int numElements, float* vector, float *result) {
    extern __shared__ float shared_mem[];
    int thread_id = threadIdx.x;
    int pos = blockDim.x * blockIdx.x + thread_id;

    if (pos < numElements) {
        shared_mem[thread_id] = vector[pos];
    }

    for (int k = blockDim.x >> 1; k >= 1; k >>= 1) {
        __syncthreads();
        if (thread_id < k && thread_id + k < numElements) {
            shared_mem[thread_id] += shared_mem[thread_id + k];
        }
    }

    if (thread_id == 0) {
        result[blockIdx.x] = shared_mem[0];
    }

}