#include <iostream>
#include <CosineVector.cuh>


int main() {
    int n = 2;
    int vector_size = n * sizeof(float);

    float* vec1 = (float*) malloc(vector_size);
    float* vec2 = (float*) malloc(vector_size);

    vec1[0] = 2;
    vec1[1] = 2;
    vec2[0] = 0;
    vec2[1] = 1;

    float ans = CosineVector(n, vec1, vec2);

    std::cout << ans;

    return 0;
}

