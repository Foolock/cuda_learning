// reduction.cu
#include <iostream>
#include <vector>
#include <numeric>
#include <omp.h>

__global__ void cuda_reduction(int *input, int *output, int N) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int cpu_reduction_openmp(const std::vector<int>& data) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    return sum;
}

int main() {
    const int N = 1 << 20; // 1M elements
    std::vector<int> h_data(N, 1); // initialize with all ones

    // CPU OpenMP reduction
    int cpu_sum = cpu_reduction_openmp(h_data);
    std::cout << "CPU (OpenMP) sum: " << cpu_sum << std::endl;

    // CUDA reduction
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMalloc(&d_output, blocks * sizeof(int));

    cuda_reduction<<<blocks, threads, threads * sizeof(int)>>>(d_input, d_output, N);

    // Now d_output has "blocks" number of partial sums
    std::vector<int> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_output, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    int gpu_sum = std::accumulate(h_partial.begin(), h_partial.end(), 0);
    std::cout << "GPU (CUDA) sum: " << gpu_sum << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

