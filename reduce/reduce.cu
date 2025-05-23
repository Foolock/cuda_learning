// reduce.cu
#include <iostream>
#include <vector>
#include <numeric>
#include <omp.h>
#include <chrono>

#define BLOCK_SIZE 128 
#define NUM_ITERS_PER_THREAD 4

// handle errors in CUDA call
#define CUDACHECK(call)                                                        \
{                                                                          \
   const cudaError_t error = call;                                         \
   if (error != cudaSuccess)                                               \
   {                                                                       \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
      exit(1);                                                             \
   }                                                                       \
} (void)0  // Ensures a semicolon is required after the macro call.

void cpu_reduction_openmp(const std::vector<int>& data) {
    int sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "cpu runtime (openmp): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";
    std::cout << "cpu (openmp) sum: " << sum << std::endl;
}

void thread_index_check() {

  size_t block_size = 128;

  std::cout << "check thread index under block_size = " << block_size << "\n";

  std::cout << "thread index for reduction #1\n";
  for(unsigned int s = 1; s < block_size; s *= 2) {
    std::cout << "level : " << s << "\n";
    for(unsigned int tid = 0; tid < block_size; tid++) {
      if(tid % (2 * s) == 0) {
        std::cout << "tid = " << tid << ", tid + s = " << tid + s << "\n";
      }
    }
  }
}

// CUDA reduction version #1: interleaved addressing
// problem: highly divergent warps, and % operator is very slow
__global__ void reduce1(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // each thread loads one element from global memory to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < N)? g_idata[i] : 0;
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  for(unsigned int s = 1; s < blockDim.x; s *= 2) {
    if(tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver1(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    int grid_size = (num_element_left + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    reduce1<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver1, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver1) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

// CUDA reduction version #2: use strided index and non-divergent branch 
// problem: bank conflict 
__global__ void reduce2(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // each thread loads one element from global memory to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < N)? g_idata[i] : 0;
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  for(unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if(index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver2(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    int grid_size = (num_element_left + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    reduce2<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver2, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver2) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

// CUDA reduction version #3: use sequential addressing to remove bank conflict 
__global__ void reduce3(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // each thread loads one element from global memory to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < N)? g_idata[i] : 0;
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver3(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    int grid_size = (num_element_left + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    reduce3<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver3, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver3) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

// CUDA reduction version #4: perform add during shared memory load 
__global__ void reduce4(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // perform first level of reduction by loading two element from global memory 
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // ×2
  if (i < N) {
    sdata[tid] = g_idata[i] + ((i + blockDim.x) < N ? g_idata[i + blockDim.x] : 0);
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver4(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    // halve the number of blocks launched
    // this is done by saying now one block can handle the work that is originally handled by two blocks in #3
    int grid_size = (num_element_left + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);; 
    reduce4<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver4, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver4) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

// CUDA reduction version #5: unroll the last warp to remove __syncthreads
// volatile ensures correct memory order (prevent compiler from reorder instructions when reading/writing sdata)  
__device__ void warpReduce(volatile int* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void reduce5(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // perform first level of reduction by loading two element from global memory 
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // ×2
  if (i < N) {
    sdata[tid] = g_idata[i] + ((i + blockDim.x) < N ? g_idata[i + blockDim.x] : 0);
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if(tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if(tid < 32) warpReduce(sdata, tid);

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver5(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    // halve the number of blocks launched
    // this is done by saying now one block can handle the work that is originally handled by two blocks in #3
    int grid_size = (num_element_left + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);; 
    reduce5<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver5, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver5) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

// CUDA reduction version #6: completely unroll for different block size
template <unsigned int blockSize>
__device__ void warpReduce1(volatile int* sdata, int tid) {
  if constexpr (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if constexpr (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if constexpr (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if constexpr (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if constexpr (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if constexpr (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // perform first level of reduction by loading two element from global memory 
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // ×2
  if (i < N) {
    sdata[tid] = g_idata[i] + ((i + blockDim.x) < N ? g_idata[i + blockDim.x] : 0);
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  if constexpr (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if(tid < 32) warpReduce1<blockSize>(sdata, tid);

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver6(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    // halve the number of blocks launched
    // this is done by saying now one block can handle the work that is originally handled by two blocks in #3
    int grid_size = (num_element_left + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);; 
    switch(BLOCK_SIZE) {
      case 512:
        reduce6<512><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);
      case 256:
        reduce6<256><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);
      case 128:
        reduce6<128><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);
    }

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver6, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver6) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

// CUDA reduction version #6+: completely unroll for different block size
// also make NUM_ITERS_PER_THREAD as template parameter
// NUM_ITERS_PER_THREAD: the number of element pre-added per thread during shared memory load  
template <unsigned int blockSize, unsigned int num_iter_per_thread>
__global__ void reduce6_plus(int *g_idata, int *g_odata, int N) {
  
  extern __shared__ int sdata[];

  // perform first level of reduction by loading num_iter_per_thread element from global memory 
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * (blockDim.x * num_iter_per_thread) + threadIdx.x; // ×2
  int sum = 0;
  #pragma unroll
  for (int i = 0; i < num_iter_per_thread; ++i) {
    if (idx + i * blockDim.x < N) {
      sum += g_idata[idx + i * blockDim.x];
    }
  }
  sdata[tid] = sum;
  __syncthreads();

  // do reduction in shared memory
  // s is stride, saying the distance between the elements one thread handles
  if constexpr (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if(tid < 32) warpReduce1<blockSize>(sdata, tid);

  // write result for this block to global memory
  if(tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
  }

}

void gpu_reduction_ver6_plus(const std::vector<int>& h_data) {

  const int N = h_data.size();

  int *g_idata, *g_odata;
  CUDACHECK(cudaMalloc(&g_idata, sizeof(int) * N));
  // g_odata's size needs to be at least (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
  // to store the partial sums from the first step
  CUDACHECK(cudaMalloc(&g_odata, sizeof(int) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)));
  CUDACHECK(cudaMemcpyAsync(g_idata, h_data.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

  // recursively invoke kernel
  int num_element_left = N;
  int *input = g_idata;
  int *output = g_odata;

  auto start = std::chrono::high_resolution_clock::now();
  while(num_element_left > 1) {
    // halve the number of blocks launched
    // this is done by saying now one block can handle the work that is originally handled by two blocks in #3
    int grid_size = (num_element_left + (BLOCK_SIZE * NUM_ITERS_PER_THREAD - 1)) / (BLOCK_SIZE * NUM_ITERS_PER_THREAD);; 
    switch(BLOCK_SIZE) {
      case 512:
        reduce6_plus<512, NUM_ITERS_PER_THREAD><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);
      case 256:
        reduce6_plus<256, NUM_ITERS_PER_THREAD><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);
      case 128:
        reduce6_plus<128, NUM_ITERS_PER_THREAD><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, num_element_left);
    }

    num_element_left = grid_size;
    int *temp = input;
    input = output;
    output = temp;
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "gpu runtime (ver6+, no memcpy): " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << " us\n";

  int gpu_sum = 0;
  CUDACHECK(cudaMemcpyAsync(&gpu_sum, input, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "gpu (ver6+) sum: " << gpu_sum << "\n";

  CUDACHECK(cudaFree(g_idata));
  CUDACHECK(cudaFree(g_odata));

}

int main() {
    size_t step_size = 30;
    const int N = 1 << step_size; // 1M elements
    std::vector<int> h_data(N, 1); // initialize with all ones
    std::cout << "number of elements = " << N << "\n";

    // CPU openmp reduction
    cpu_reduction_openmp(h_data);

    // GPU reduction with thread divergence
    gpu_reduction_ver1(h_data);

    // GPU reduction with bank conflict 
    gpu_reduction_ver2(h_data);

    // GPU reduction with sequential addressing to remove bank conflict
    // thread idling
    gpu_reduction_ver3(h_data);

    // GPU reduction with sequential addressing to remove bank conflict
    // perform add during shared memory load
    gpu_reduction_ver4(h_data);

    // GPU reduction with sequential addressing to remove bank conflict
    // perform add during shared memory load
    // remove syncthread by unroll for loop in the last 6 iteration
    gpu_reduction_ver5(h_data);

    // GPU reduction with sequential addressing to remove bank conflict
    // perform add during shared memory load
    // completely unroll for loop 
    gpu_reduction_ver6(h_data);

    // GPU reduction with sequential addressing to remove bank conflict
    // perform add during shared memory load
    // completely unroll for loop, also added num_iter_per_thread as template parameter
    gpu_reduction_ver6_plus(h_data);

    return 0;
}


































