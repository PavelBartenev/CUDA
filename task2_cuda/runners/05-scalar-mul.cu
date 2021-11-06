#include <iostream>
#include <ScalarMulRunner.cuh>

int main() {
    int n = 20000;
    int vector_size = n * sizeof(float);

    float* vec1 = (float*) malloc(vector_size);
    float* vec2 = (float*) malloc(vector_size);

    for (int i = 0; i < n; ++i) {
        vec1[i] = 3;
        vec2[i] = 2;
    }

    float ans = ScalarMulSumPlusReduction(n, vec1, vec2, 512);

    return 0;
}
