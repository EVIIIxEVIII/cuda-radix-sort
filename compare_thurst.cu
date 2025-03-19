#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 100000000

int main() {
    // Allocate host and device memory
    std::vector<int> h_data(N);
    for (int i = 0; i < N; i++) h_data[i] = rand();

    // Copy data to GPU
    thrust::device_vector<int> d_data = h_data;

    // Measure Thrust sort time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::sort(d_data.begin(), d_data.end());
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms\n";

    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n\n");
    for (int i = 0; i < N-1; i++) {
        if(h_data[i] > h_data[i+1]) {
            printf("FAILED TO SORT THE ARRAY %d < %d", h_data[i], h_data[i+1]);
            printf("\n\n");
            return 0;
        }
    }

    printf("\n\nSUCCESS\n\n");

    return 0;
}
