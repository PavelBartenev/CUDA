#include <ScalarMul.cuh>

/*
 * Calculates scalar multiplication for block
 */
__global__
void ScalarMulBlock(int numElements, float* vector1, float* vector2, float *result) {
    extern __shared__ float shared_mem[];
    int thread_id = threadIdx.x;
    int pos = blockDim.x * blockIdx.x + thread_id;

    if (pos < numElements) {
        shared_mem[thread_id] = vector1[pos] * vector2[pos];
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
