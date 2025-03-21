#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 32
#define N 134217728
#define RADIX 2

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

double getTimeMicroseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

void init_array(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand();
    }
}

__global__ void compute_global_count(int* arr, int* global_count, int n, int iter) {
    __shared__ int local_count_map[4];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (tid < 4) local_count_map[tid] = 0;

    __syncthreads();

    int digit = 0;
    int active = (gid < n);
    if (active) {
        digit = (arr[gid] >> (2 * iter)) & 3;
    }

    int mask_0 = __ballot_sync(0xFFFFFFFF, digit == 0);
    int mask_1 = __ballot_sync(0xFFFFFFFF, digit == 1);
    int mask_2 = __ballot_sync(0xFFFFFFFF, digit == 2);
    int mask_3 = __ballot_sync(0xFFFFFFFF, digit == 3);

    int count_0 = __popc(mask_0);
    int count_1 = __popc(mask_1);
    int count_2 = __popc(mask_2);
    int count_3 = __popc(mask_3);

    if (tid == 0) {
        local_count_map[0] += count_0;
        local_count_map[1] += count_1;
        local_count_map[2] += count_2;
        local_count_map[3] += count_3;
    }

    __syncthreads();

    if (tid < 4) {
        global_count[gridDim.x * tid + blockIdx.x] = local_count_map[tid];
    }
}
__global__ void in_lane_scan(int* arr, int* in_lane_scans, int* sums, int n) {
    __shared__ int local_offset[BLOCK_SIZE];
    __shared__ int temp[BLOCK_SIZE];
    __shared__ int sum;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) sum = 0;
    if (gid < n) {
        temp[tid] = (tid == 0) ? 0 : arr[gid - 1];
    }

    if (gid < n) {
        atomicAdd(&sum, arr[gid]);
    }

    __syncthreads();

    if (tid < BLOCK_SIZE) {
        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
            int val = (tid >= offset) ? temp[tid - offset] : 0;
            __syncthreads();
            temp[tid] += val;
            __syncthreads();
        }

        local_offset[tid] = temp[tid];
    }

    __syncthreads();

    if (gid < n) {
        in_lane_scans[gid] = local_offset[tid];
    }

    if (tid == 0) {
        sums[bid] = sum;
    }
}

__global__ void parallel_scan_sums(int* sums, int* temp, int sums_num) {


}

__global__ void in_lane_propagation(int* in_lane_scans, int* sums, int* res, int n) {
    const int bid = blockIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        res[gid] = in_lane_scans[gid] + sums[bid];
    }
}


__global__ void radix_sort(int* arr, int* res, int* global_offset, int n, int iter) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        int digit = (arr[gid] >> (2 * iter)) & 3;
        int final_idx = atomicAdd(&global_offset[digit * gridDim.x + blockIdx.x], 1);
        res[final_idx] = arr[gid];
    }
}

int main() {
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int const maps_size = 4 * gridDim.x;

    dim3 blockDimILS(BLOCK_SIZE);
    dim3 gridDimILS((maps_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Threads num (radix): %d \n", blockDim.x * gridDim.x);
    printf("Blocks num  (radix): %d \n", gridDim.x);

    printf("Threads num (ils): %d \n", blockDimILS.x * gridDimILS.x);
    printf("Blocks num  (ils): %d \n", gridDimILS.x);

    int* h_arr = (int*)malloc(N * sizeof(int));
    int* h_res = (int*)malloc(N * sizeof(int));
    int* h_global_count = (int*)malloc(maps_size * sizeof(int));
    int* h_global_offset = (int*)malloc(maps_size * sizeof(int));

    int* h_sums = (int*)malloc(gridDim.x * sizeof(int));
    int* h_sums_offsets = (int*)malloc(gridDim.x * sizeof(int));

    init_array(h_arr, N);

    int* d_arr;
    int* d_res;
    int* d_global_count;
    int* d_global_offset;
    int* d_sums;
    int* d_in_lane_scans;

    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_res, N * sizeof(int));
    cudaMalloc(&d_global_count,  maps_size * sizeof(int));
    cudaMalloc(&d_global_offset, maps_size * sizeof(int));
    cudaMalloc(&d_in_lane_scans, maps_size * sizeof(int));
    cudaMalloc(&d_sums, gridDim.x * sizeof(int));

    cudaMemset(d_global_offset, 0, maps_size * sizeof(int));
    cudaMemset(d_global_count,  0, maps_size * sizeof(int));
    cudaMemset(d_in_lane_scans, 0, maps_size * sizeof(int));
    cudaMemset(d_sums,          0, gridDim.x * sizeof(int));

    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    double totalGc = 0;
    double totalIls = 0;
    double totalIlp = 0;
    double totalCpu = 0;
    double totalRadix = 0;

    cudaFree(0);
    double start = getTimeMicroseconds();
    for (int i = 0; i < 16; i++) {
        double gcStart = getTimeMicroseconds();
        compute_global_count<<<gridDim, blockDim>>>(d_arr, d_global_count, N, i);
        CHECK_CUDA(cudaDeviceSynchronize());
        double gcEnd = getTimeMicroseconds();

        double ilsStart = getTimeMicroseconds();
        in_lane_scan<<<gridDimILS, blockDimILS>>>(d_global_count, d_in_lane_scans, d_sums, maps_size);
        CHECK_CUDA(cudaDeviceSynchronize());
        double ilsEnd = getTimeMicroseconds();

        double startCpu = getTimeMicroseconds();
        cudaMemcpy(h_sums, d_sums, gridDim.x * sizeof(int), cudaMemcpyDeviceToHost);
        h_sums_offsets[0] = 0;
        for (int j = 1; j < gridDim.x; j++) {
            h_sums_offsets[j] = h_sums_offsets[j - 1] + h_sums[j - 1];
        }
        cudaMemcpy(d_sums, h_sums_offsets, gridDim.x * sizeof(int), cudaMemcpyHostToDevice);
        double endCpu = getTimeMicroseconds();

        double ilpStart = getTimeMicroseconds();
        in_lane_propagation<<<gridDimILS, blockDimILS>>>(d_in_lane_scans, d_sums, d_global_offset, maps_size);
        CHECK_CUDA(cudaDeviceSynchronize());
        double ilpEnd = getTimeMicroseconds();

        double radixStart = getTimeMicroseconds();
        radix_sort<<<gridDim, blockDim>>>(d_arr, d_res, d_global_offset, N, i);
        CHECK_CUDA(cudaDeviceSynchronize());
        double radixEnd = getTimeMicroseconds();

        cudaMemcpy(d_arr, d_res, N * sizeof(int), cudaMemcpyDeviceToDevice);

        totalGc += (gcEnd - gcStart);
        totalIls += (ilsEnd - ilsStart);
        totalIlp += (ilpEnd - ilpStart);
        totalCpu += (endCpu - startCpu);
        totalRadix += (radixEnd - radixStart);
    }
    double end = getTimeMicroseconds();

    printf("\n\nTime taken (total): %lf microseconds\n\n", end - start);

    printf("Time taken (gc)   : %lf microseconds\n", totalGc);
    printf("Time taken (ils)  : %lf microseconds\n", totalIls);
    printf("Time taken (ilp)  : %lf microseconds\n", totalIlp);
    printf("Time taken (cpu)  : %lf microseconds\n", totalCpu);
    printf("Time taken (radix): %lf microseconds\n", totalRadix);

    cudaMemcpy(h_res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_count, d_global_count, maps_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_offset, d_global_offset, maps_size * sizeof(int), cudaMemcpyDeviceToHost);

    //printf("\nSorted array: \n");
    //for (int i = 0; i < N; i++) {
    //    printf("%d \n", h_res[i]);
    //}

    printf("\n\n");
    for (int i = 0; i < N-1; i++) {
        if(h_res[i] > h_res[i+1]) {
            printf("FAILED TO SORT THE ARRAY %d < %d", h_res[i], h_res[i+1]);
            printf("\n\n");
            return 0;
        }
    }

    printf("\n\nTEST PASSED! SORTING PERFORMED CORRECTLY!\n\n");

    free(h_arr);
    free(h_res);

    cudaFree(d_arr);
    cudaFree(d_res);
    cudaFree(d_global_offset);
    cudaFree(d_global_count);
}

