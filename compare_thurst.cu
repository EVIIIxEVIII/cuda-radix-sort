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
    auto start = std::chrono::high_resolution_clock::now();
    thrust::sort(d_data.begin(), d_data.end());
    auto end = std::chrono::high_resolution_clock::now();

    float thrust_time = std::chrono::duration<float>(end - start).count();
    std::cout << "Thrust GPU sort time: " << thrust_time << " seconds" << std::endl;

    return 0;
}
