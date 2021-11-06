#include <MatrixVectorMul.cuh>


__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_mem[];

    float res_i = 0.0;
    int blocks_in_vec = ((width + blockDim.x - 1) / blockDim.x);

    for (int block_index = 0; block_index < blocks_in_vec; ++block_index) {
        if ((block_index * blockDim.x) + threadIdx.x < width) {
            shared_mem[threadIdx.x] = vector[threadIdx.x + block_index * blockDim.x];
        } else {
            shared_mem[threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int elem_id = 0; elem_id < blockDim.x; ++elem_id) {
            res_i += matrix[thread_id + height * (elem_id + blockDim.x * block_index)] * shared_mem[elem_id];
        }

        __syncthreads();
    }

    if (thread_id < height) {
        result[thread_id] = res_i;
    }
}
