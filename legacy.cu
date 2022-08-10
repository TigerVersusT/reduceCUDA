#include <iostream>
#include <vector>

// use one dimensional grid and one dimensional block
__global__ void reduceSum(float *input, float *result, const int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= size) {
    return;
  }

  int len = blockDim.x / 2;
  while (len > 0) {
    if (threadIdx.x < len) {
      input[x] += input[x + len];
    }

    len /= 2;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result[blockIdx.x] = input[blockIdx.x];
  }
}

// use shared memory
//  datasize 10*3*1024*1024
//  Duration                usecond                           744.58
//  SOL DRAM                %                                 26.22
//  SOL L1/TEX Cache        %                                 94.54
//  SOL L2 Cache            %                                 9.81
//  SM [%]                  %                                 75.04
__global__ void reduceSumSharedMemory(float *input, float *result,
                                      const int size) {
  extern __shared__ float tempArray[];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= size) {
    return;
  }

  // prepare data
  tempArray[threadIdx.x] = input[x];
  __syncthreads();

  int len = blockDim.x / 2;
  while (len > 0) {
    if (threadIdx.x < len) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + len];
    }

    len /= 2;
    __syncthreads();
  }

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
  __syncthreads();
}

// loads two data at initial stage
//  datasize 10*3*1024*1024
//  Duration                usecond                           372.22
//  SOL DRAM                %                                 52.08
//  SOL L1/TEX Cache        %                                 91.3
//  SOL L2 Cache            %                                 19.09
//  SM [%]                  %                                 73.88
__global__ void reduceSumTwoLoads(float *input, float *result, const int size) {
  extern __shared__ float tempArray[];

  int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (x >= size) {
    return;
  }

  // prepare data
  tempArray[threadIdx.x] = input[x] + input[x + blockDim.x];
  __syncthreads();

  int len = blockDim.x / 2;
  while (len > 0) {
    if (threadIdx.x < len) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + len];
    }

    len /= 2;
    __syncthreads();
  }

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

__device__ void wrapReduce(volatile float *input, int id) {
  input[id] += input[id + 32];
  input[id] += input[id + 16];
  input[id] += input[id + 8];
  input[id] += input[id + 4];
  input[id] += input[id + 2];
  input[id] += input[id + 1];
}

//  unroll last wrap
//  datasize 10*3*1024*1024
//  Duration                usecond                           247.52
//  SOL DRAM                %                                 78.40
//  SOL L1/TEX Cache        %                                 59.03
//  SOL L2 Cache            %                                 28.79
//  SM [%]                  %                                 52.44
__global__ void reduceSumUnrollLastWrap(float *input, float *result, int size) {
  extern __shared__ float tempArray[];

  int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (x >= size) {
    return;
  }

  // prepare data
  tempArray[threadIdx.x] = input[x] + input[x + blockDim.x];
  __syncthreads();

  int len = blockDim.x / 2;
  while (len > 32) {
    if (threadIdx.x < len) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + len];
    }

    len /= 2;
    __syncthreads();
  }

  // last wrap
  if (threadIdx.x < 32) {
    wrapReduce(tempArray, threadIdx.x);
  }

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

// manually unroll iteration, assume blocksize as 512
//  datasize 10*3*1024*1024
//  Duration                usecond                           300.45
//  SOL DRAM                %                                 53.72
//  SOL L1/TEX Cache        %                                 23.08
//  SOL L2 Cache            %                                 28.79
//  SM [%]                  %                                 51.03
__global__ void reduceSumUnroll512(float *input, float *result, int size) {
  extern __shared__ float tempArray[];

  int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (x >= size) {
    return;
  }

  // prepare data
  tempArray[threadIdx.x] = input[x] + input[x + blockDim.x];
  __syncthreads();

  // manually unroll all 512 threads
  if (threadIdx.x < 256) {
    tempArray[threadIdx.x] += tempArray[threadIdx.x + 256];
  }
  __syncthreads();

  if (threadIdx.x < 128) {
    tempArray[threadIdx.x] += tempArray[threadIdx.x + 128];
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    tempArray[threadIdx.x] += tempArray[threadIdx.x + 64];
  }
  __syncthreads();

  // last wrap
  if (threadIdx.x < 32) {
    wrapReduce(tempArray, threadIdx.x);
  }

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

// manually unroll iteration, set blockSize to 256
//  datasize 10*3*1024*1024
//  Duration                usecond                           272.86
//  SOL DRAM                %                                 71.11
//  SOL L1/TEX Cache        %                                 53.96
//  SOL L2 Cache            %                                 25.69
//  SM [%]                  %                                 49.50
__global__ void reduceSumUnroll256(float *input, float *result, int size) {
  extern __shared__ float tempArray[];

  int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (x >= size) {
    return;
  }

  // prepare data
  tempArray[threadIdx.x] = input[x] + input[x + blockDim.x];
  __syncthreads();

  // manually unroll all 256 threads
  if (threadIdx.x < 128) {
    tempArray[threadIdx.x] += tempArray[threadIdx.x + 128];
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    tempArray[threadIdx.x] += tempArray[threadIdx.x + 64];
  }
  __syncthreads();

  // last wrap
  if (threadIdx.x < 32) {
    wrapReduce(tempArray, threadIdx.x);
  }

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

// manually unroll iteration, set blockSize to 128, sum up 256 elements per
// block
//  datasize 10*3*1024*1024
//  Duration                usecond                           239.42
//  SOL DRAM                %                                 80.80
//  SOL L1/TEX Cache        %                                 60.81
//  SOL L2 Cache            %                                 29.63
//  SM [%]                  %                                 54.01
__global__ void reduceSumUnroll128(float *input, float *result) {
  __shared__ float tempArray[128];

  int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // prepare data
  tempArray[threadIdx.x] = input[x] + input[x + blockDim.x];
  __syncthreads();

  // manually unroll all 128 threads
  if (threadIdx.x < 64) {
    tempArray[threadIdx.x] += tempArray[threadIdx.x + 64];
  }
  __syncthreads();

  // last wrap
  if (threadIdx.x < 32) {
    wrapReduce(tempArray, threadIdx.x);
  }

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

__global__ void reduceSumLast256(float *input, float *result, int size) {
  __shared__ float tempArray[256];
  int x = threadIdx.x;

  // prepare data, the unused element should be zero to fit original logics
  if (x < size) {
    tempArray[x] = input[x];
  } else {
    tempArray[x] = 0;
  }
  __syncthreads();

  if (x < 128) {
    tempArray[x] += tempArray[x + 128];
  }
  __syncthreads();

  if (x < 64) {
    tempArray[x] += tempArray[x + 64];
  }
  __syncthreads();

  // last wrap
  if (x < 32) {
    wrapReduce(tempArray, x);
  }

  // write internal result back to global memory
  if (x == 0) {
    *result += tempArray[x];
  }
}

void testReduceSumTwoLoads(float *input, float *result, int size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  int internalSize = size;
  // for (int internalSize = size; internalSize > 1; internalSize /= 128) {
  dim3 block(128, 1);
  dim3 grid((internalSize + 128 - 1) / 128 / 2, 1);

  // std::cout << "internalSize: " << internalSize << std::endl;

  reduceSumTwoLoads<<<grid, block, sizeof(float) * 128>>>(d_input, d_input,
                                                          internalSize);
  //}

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}

void testReduceSumUnrollLastWrap(float *input, float *result, int size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  int internalSize = size;
  // for (int internalSize = size; internalSize > 1; internalSize /= 128) {
  dim3 block(128, 1);
  dim3 grid((internalSize + 128 - 1) / 128 / 2, 1);

  // std::cout << "internalSize: " << internalSize << std::endl;

  reduceSumUnrollLastWrap<<<grid, block, sizeof(float) * 128>>>(
      d_input, d_input, internalSize);
  //}

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}

void testReduceSumUnroll512(float *input, float *result, int size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  int internalSize = size;
  // for (int internalSize = size; internalSize > 1; internalSize /= 128) {
  dim3 block(512, 1);
  dim3 grid((internalSize + 512 - 1) / 512 / 2, 1);

  // std::cout << "internalSize: " << internalSize << std::endl;

  reduceSumUnroll512<<<grid, block, sizeof(float) * 512>>>(d_input, d_input,
                                                           internalSize);
  //}

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}

void testReduceSumUnroll256(float *input, float *result, int size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  int internalSize = size;
  // for (int internalSize = size; internalSize > 1; internalSize /= 128) {
  dim3 block(256, 1);
  dim3 grid((internalSize + 256 - 1) / 256 / 2, 1);

  // std::cout << "internalSize: " << internalSize << std::endl;

  reduceSumUnroll256<<<grid, block, sizeof(float) * 256>>>(d_input, d_input,
                                                           internalSize);
  //}

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}

void testReduceSumUnroll128(float *input, float *result, size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 256;
    internalSize = internalSize - remain;

    // std::cout << "internalSize: " << internalSize << std::endl;
    // std::cout << "remaimn: " << remain << std::endl;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 2;
    dim3 grid(blockNumber, 1);

    // std::cout << "blockNumber: " << blockNumber << std::endl;

    if (blockNumber > 0) {
      reduceSumUnroll128<<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(256, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceSumLast256<<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceSumLast256<<<grid1, block1>>>(d_input + (blockNumber * 128 * 2),
                                            d_input, remain);
      }
    }

    internalSize = blockNumber;
    if (internalSize <= 1) {
      break;
    }
  }

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}

void debugReduceLast256(float *input, float *result, int size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  dim3 block(256, 1);
  dim3 grid(1, 1);
  reduceSumLast256<<<grid, block>>>(d_input + 1, d_input, 254);

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
}

void testReduceSum() {
  const int size = 32 * 1024 * 1024;
  std::vector<float> h_input(size, 0);
  srand((unsigned)time(NULL));
  for (int i = 0; i < size; i++) {
    h_input[i] = 1;
  }

  dim3 block1(128, 1);
  dim3 grid1((size + block1.x - 1) / block1.x, 1);
  int internalSize = grid1.x;
  // device memory
  float *d_input, *d_output1, *d_output2, h_output;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMalloc(&d_output1, sizeof(float) * internalSize);
  cudaMalloc(&d_output2, sizeof(float));
  cudaMemcpy(d_input, h_input.data(), sizeof(float) * size,
             cudaMemcpyHostToDevice);

  reduceSum<<<grid1, block1>>>(d_input, d_output1, size);

  dim3 block2(internalSize, 1);
  dim3 grid2(1, 1);
  reduceSum<<<grid2, block2>>>(d_output1, d_output2, internalSize);

  // reduceSumSharedMemory<<<grid, block, sizeof(float) * 128>>>(d_input,
  // d_output,
  //                                                            size);
  cudaMemcpy(&h_output, d_output2, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output1);
  cudaFree(d_output2);

  std::cout << "gpu result: " << h_output << std::endl;
}

void testReduceSumSharedMemory(float *input, float *result, int size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  int internalSize = size;
  // for (int internalSize = size; internalSize >= 1; internalSize /= 128) {
  dim3 block(128, 1);
  dim3 grid((internalSize + 128 - 1) / 128, 1);

  std::cout << "internalSize: " << internalSize << std::endl;

  reduceSumSharedMemory<<<grid, block, sizeof(float) * 128>>>(d_input, d_input,
                                                              internalSize);
  //}

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}