#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define N 73
#define RADIX 2

void init_array(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }
}

__global__ void compute_global_count(int* arr, int* global_count, int n, int iter) {
    __shared__ int local_count_map[4];
    __shared__ int local_offset_map[4];

    __shared__ int digits[32];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (tid < 4) {
        local_count_map[tid] = 0;
        local_offset_map[tid] = 0;
    }
    __syncthreads();

    if (gid < n) {
        digits[tid] = (arr[gid] >> (2 * iter)) & 3;
        atomicAdd(&local_count_map[digits[tid]], 1);
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < 4; i++) {
            local_offset_map[i] = local_offset_map[i - 1] + local_count_map[i - 1];
        }
    }
    __syncthreads();

    if (tid < 4) {
        global_count[gridDim.x * tid + blockIdx.x] = local_count_map[tid];
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

    int* h_arr = (int*)malloc(N * sizeof(int));
    int* h_res = (int*)malloc(N * sizeof(int));
    int* h_global_count = (int*)malloc(gridDim.x * pow(RADIX, 2) * sizeof(int));
    int* h_global_offset = (int*)malloc(gridDim.x * pow(RADIX, 2) * sizeof(int));

    init_array(h_arr, N);

    int* d_arr;
    int* d_res;
    int* d_global_count;
    int* d_global_offset;

    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_res, N * sizeof(int));
    cudaMalloc(&d_global_count,  gridDim.x * pow(RADIX, 2) * sizeof(int));
    cudaMalloc(&d_global_offset, gridDim.x * pow(RADIX, 2) * sizeof(int));

    cudaMemset(d_global_offset, 0, gridDim.x * pow(RADIX, 2) * sizeof(int));
    cudaMemset(d_global_count,  0, gridDim.x * pow(RADIX, 2) * sizeof(int));

    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < 7; i++) {
        compute_global_count<<<gridDim, blockDim>>>(d_arr, d_global_count, N, i);
        cudaDeviceSynchronize();
        cudaMemcpy(h_global_count, d_global_count, gridDim.x * pow(RADIX, 2) * sizeof(int), cudaMemcpyDeviceToHost);

        h_global_offset[0] = 0;
        for (int j = 1; j < 4 * gridDim.x; j++) {
            h_global_offset[j] = h_global_offset[j - 1] + h_global_count[j - 1];
        }

        printf("Global offset: \n");
        for(int j = 0; j < 4 * gridDim.x; j++) {
            printf("%d ", h_global_offset[j]);
        }

        cudaMemcpy(d_global_offset, h_global_offset, gridDim.x * pow(RADIX, 2) * sizeof(int), cudaMemcpyHostToDevice);
        radix_sort<<<gridDim, blockDim>>>(d_arr, d_res, d_global_offset, N, i);
        cudaDeviceSynchronize();

        cudaMemcpy(d_arr, d_res, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_res, d_res, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_count, d_global_count, gridDim.x * pow(RADIX, 2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_global_offset, d_global_offset, gridDim.x * pow(RADIX, 2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("\nSorted array: \n");
    for (int i = 0; i < N; i++) {
        printf("%d \n", h_res[i]);
    }

    printf("\n\n");
    for (int i = 0; i < N-1; i++) {
        if(h_res[i] > h_res[i+1]) {
            printf("FAILED TO SORT THE ARRAY %d < %d", h_res[i], h_res[i+1]);
            return 0;
        }
    }
    printf("SUCCESS");
    printf("\n\n");


    free(h_arr);
    free(h_res);

    cudaFree(d_arr);
    cudaFree(d_res);
    cudaFree(d_global_offset);
    cudaFree(d_global_count);
}

