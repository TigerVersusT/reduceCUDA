

template <int threadNumber>
__device__ void wrapReduceByShuffle(volatile float *input, int id) {
  if (threadNumber >= 64) input[id] += input[id + 32];

  // read data into thread register
  int sum = input[id];

  // Use XOR mode to perform butterfly reduction
  sum += __shfl_xor_sync(0xffffffff, sum, 16, 32);
  sum += __shfl_xor_sync(0xffffffff, sum, 8, 32);
  sum += __shfl_xor_sync(0xffffffff, sum, 4, 32);
  sum += __shfl_xor_sync(0xffffffff, sum, 2, 32);
  sum += __shfl_xor_sync(0xffffffff, sum, 1, 32);

  // write register value to shared memory
  if (id == 0) *input = sum;
}

template <int threadNumber>
__device__ void wrapReduce(volatile float *input, int id) {
  if (threadNumber >= 64) input[id] += input[id + 32];
  if (threadNumber >= 32) input[id] += input[id + 16];
  if (threadNumber >= 16) input[id] += input[id + 8];
  if (threadNumber >= 8) input[id] += input[id + 4];
  if (threadNumber >= 4) input[id] += input[id + 2];
  if (threadNumber >= 2) input[id] += input[id + 1];
}

template <int threadNumber>
__device__ void reduceOnSharedMempry(float *tempArray, int id) {
  // manually unroll max to 1024 threads

  if (threadNumber >= 1024) {
    if (threadIdx.x < 512) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + 512];
    }
    __syncthreads();
  }

  if (threadNumber >= 512) {
    if (threadIdx.x < 256) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + 256];
    }
    __syncthreads();
  }

  if (threadNumber >= 256) {
    if (threadIdx.x < 128) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + 128];
    }
    __syncthreads();
  }

  if (threadNumber >= 128) {
    if (threadIdx.x < 64) {
      tempArray[threadIdx.x] += tempArray[threadIdx.x + 64];
    }
    __syncthreads();
  }

  // last wrap
  if (threadIdx.x < 32) wrapReduceByShuffle<128>(tempArray, threadIdx.x);
}

template <int lastNumber>
__global__ void reduceLast(float *input, float *result, int size) {
  __shared__ float tempArray[lastNumber];
  int x = threadIdx.x;

  // prepare data, the unused element should be zero to fit original logics
  if (x < size) {
    tempArray[x] = input[x];
  } else {
    tempArray[x] = 0;
  }
  __syncthreads();

  reduceOnSharedMempry<lastNumber>(tempArray, x);

  // write internal result back to global memory
  if (x == 0) {
    *result += tempArray[x];
  }
}

__global__ void reduceSumByWrap(float *input, float *result) {
  __shared__ float tempArray[32];

  int wrapNum = blockDim.x / 32;
  int wrapId = threadIdx.x >> 5;
  int laneId = threadIdx.x & 31;
  volatile float *basePtr = input + wrapId * 32;

  // read data into registers and perform wrap reduce
  float value = *(basePtr + laneId);
  value += __shfl_xor_sync(0xffffffff, value, 16, 32);
  value += __shfl_xor_sync(0xffffffff, value, 8, 32);
  value += __shfl_xor_sync(0xffffffff, value, 4, 32);
  value += __shfl_xor_sync(0xffffffff, value, 2, 32);
  value += __shfl_xor_sync(0xffffffff, value, 1, 32);
  if (laneId == 0) {
    tempArray[wrapId] = value;
  }
  __syncthreads();

  // reduce on shared memory
  if (wrapId == 0) {
    if (laneId < wrapNum)
      value = tempArray[laneId];
    else
      value = 0;

    value += __shfl_xor_sync(0xffffffff, value, 16, 32);
    value += __shfl_xor_sync(0xffffffff, value, 8, 32);
    value += __shfl_xor_sync(0xffffffff, value, 4, 32);
    value += __shfl_xor_sync(0xffffffff, value, 2, 32);
    value += __shfl_xor_sync(0xffffffff, value, 1, 32);
  }
  if (laneId == 0) result[blockIdx.x] = value;
}

__global__ void reduceSumByWrapV2(float *input, float *result, size_t size) {
  __shared__ float tempArray[32];

  int wrapNum = blockDim.x / 32;
  int wrapId = threadIdx.x >> 5;
  int laneId = threadIdx.x & 31;

  float value = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    value += input[i];
  }

  // read data into registers and perform wrap reduce
  value += __shfl_xor_sync(0xffffffff, value, 16, 32);
  value += __shfl_xor_sync(0xffffffff, value, 8, 32);
  value += __shfl_xor_sync(0xffffffff, value, 4, 32);
  value += __shfl_xor_sync(0xffffffff, value, 2, 32);
  value += __shfl_xor_sync(0xffffffff, value, 1, 32);
  if (laneId == 0) {
    tempArray[wrapId] = value;
  }
  __syncthreads();

  // reduce on shared memory
  if (wrapId == 0) {
    if (laneId < wrapNum)
      value = tempArray[laneId];
    else
      value = 0;

    value += __shfl_xor_sync(0xffffffff, value, 16, 32);
    value += __shfl_xor_sync(0xffffffff, value, 8, 32);
    value += __shfl_xor_sync(0xffffffff, value, 4, 32);
    value += __shfl_xor_sync(0xffffffff, value, 2, 32);
    value += __shfl_xor_sync(0xffffffff, value, 1, 32);
  }
  if (laneId == 0) result[blockIdx.x] = value;
}

__device__ void reduceSumByWrapDevice(float *input, float *result) {
  __shared__ float tempArray[32];

  int wrapNum = blockDim.x / 32;
  int wrapId = threadIdx.x >> 5;
  int laneId = threadIdx.x & 31;
  volatile float *basePtr = input + wrapId * 32;

  // read data into registers and perform wrap reduce
  float value = *(basePtr + laneId);
  value += __shfl_xor_sync(0xffffffff, value, 16, 32);
  value += __shfl_xor_sync(0xffffffff, value, 8, 32);
  value += __shfl_xor_sync(0xffffffff, value, 4, 32);
  value += __shfl_xor_sync(0xffffffff, value, 2, 32);
  value += __shfl_xor_sync(0xffffffff, value, 1, 32);
  if (laneId == 0) {
    tempArray[wrapId] = value;
  }
  __syncthreads();

  // reduce on shared memory
  if (wrapId == 0) {
    if (laneId < wrapNum)
      value = tempArray[laneId];
    else
      value = 0;

    value += __shfl_xor_sync(0xffffffff, value, 16, 32);
    value += __shfl_xor_sync(0xffffffff, value, 8, 32);
    value += __shfl_xor_sync(0xffffffff, value, 4, 32);
    value += __shfl_xor_sync(0xffffffff, value, 2, 32);
    value += __shfl_xor_sync(0xffffffff, value, 1, 32);
  }
  if (laneId == 0) result[blockIdx.x] = value;
}

// first load elements to shared memory, thenn use wrap-ordered reduce
template <int threadNumber>
__global__ void reduceSumUnrollIterationsWrapReduce(float *input,
                                                    float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  // prepare data
  tempArray[threadIdx.x] = input[x];
  __syncthreads();

  reduceSumByWrapDevice(tempArray, result);
}

template <int threadNumber>
__global__ void reduceSumUnrollIterations(float *input, float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // prepare data
  tempArray[threadIdx.x] = input[x] + input[x + blockDim.x];
  __syncthreads();

  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

template <int threadNumber>
__global__ void reduceSumUnrollAndLoads4(float *input, float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 4 + threadIdx.x;

  // loads four elements into register
  float temp = input[x] + input[x + blockDim.x];
  float temp2 = input[x + blockDim.x * 2] + input[x + blockDim.x * 3];
  // write results into shared memory
  tempArray[threadIdx.x] = temp + temp2;
  __syncthreads();

  // reduce on shared memory
  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

template <int threadNumber>
__global__ void reduceSumUnrollAndLoads4LinearAccess(float *input,
                                                     float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

  // loads four elements into register
  float temp = input[x] + input[x + 1];
  float temp2 = input[x + 2] + input[x + 3];
  // write results into shared memory
  tempArray[threadIdx.x] = temp + temp2;
  __syncthreads();

  // reduce on shared memory
  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

template <int threadNumber>
__global__ void reduceSumUnrollAndLoads8(float *input, float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // loads four elements into register
  float temp = input[x] + input[x + blockDim.x];
  float temp2 = input[x + blockDim.x * 2] + input[x + blockDim.x * 3];
  float temp3 = input[x + blockDim.x * 4] + input[x + blockDim.x * 5];
  float temp4 = input[x + blockDim.x * 6] + input[x + blockDim.x * 7];
  // write results into shared memory
  tempArray[threadIdx.x] = temp + temp2 + temp3 + temp4;
  __syncthreads();

  // reduce on shared memory
  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

template <int threadNumber>
__global__ void reduceSumUnrollAndLoads8LinearAccessMemory(float *input,
                                                           float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 8 + threadIdx.x * 8;

  // loads eight elements into register
  float temp = input[x] + input[x + 1];
  float temp2 = input[x + 2] + input[x + 3];
  float temp3 = input[x + 4] + input[x + 5];
  float temp4 = input[x + 6] + input[x + 7];
  // write results into shared memory
  tempArray[threadIdx.x] = temp + temp2 + temp3 + temp4;
  __syncthreads();

  // reduce on shared memory
  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

template <int threadNumber>
__global__ void reduceSumUnrollAndLoads8VectorLinearAccessMemory(
    float *input, float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 8 + threadIdx.x * 8;

  // loads eight elements into register
  float oprand1[4], oprand2[4];
  *(reinterpret_cast<float4 *>(oprand1)) =
      *(reinterpret_cast<float4 *>(input + x));
  float temp1 = oprand1[0] + oprand1[1] + oprand1[2] + oprand1[3];

  *(reinterpret_cast<float4 *>(oprand2)) =
      *(reinterpret_cast<float4 *>(input + x + 4));
  float temp2 = oprand2[0] + oprand2[1] + oprand2[2] + oprand2[3];

  // write results into shared memory
  tempArray[threadIdx.x] = temp1 + temp2;
  __syncthreads();

  // reduce on shared memory
  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

template <int threadNumber>
__global__ void reduceSumUnrollAndLoads16LinearAccessMemory(float *input,
                                                            float *result) {
  __shared__ float tempArray[threadNumber];
  int x = blockIdx.x * blockDim.x * 16 + threadIdx.x * 16;

  // loads 16 elements into register
  float temp = input[x] + input[x + 1];
  float temp2 = input[x + 2] + input[x + 3];
  float temp3 = input[x + 4] + input[x + 5];
  float temp4 = input[x + 6] + input[x + 7];
  float temp5 = input[x + 8] + input[x + 9];
  float temp6 = input[x + 10] + input[x + 11];
  float temp7 = input[x + 12] + input[x + 13];
  float temp8 = input[x + 14] + input[x + 15];

  // write results into shared memory
  tempArray[threadIdx.x] =
      temp + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8;
  __syncthreads();

  // reduce on shared memory
  reduceOnSharedMempry<threadNumber>(tempArray, threadIdx.x);

  // write internal result back to global memory
  if (threadIdx.x == 0) {
    result[blockIdx.x] = tempArray[threadIdx.x];
  }
}

void testReduceSumUnrollAlliterationsUseRegister(float *input, float *result,
                                                 size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 256;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollIterations<128><<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(256, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<256><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<256><<<grid1, block1>>>(d_input + (blockNumber * 128 * 2),
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

void testReduceSumUnrollAllIterationsWrapReduce(float *input, float *result,
                                                size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 512;
    internalSize = internalSize - remain;

    dim3 block(512, 1);
    size_t blockNumber = internalSize / 512;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollIterationsWrapReduce<512>
          <<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(512, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<512><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<512>
            <<<grid1, block1>>>(d_input + (blockNumber * 128), d_input, remain);
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

void testReduceSumUnrollAllIterationLoads4(float *input, float *result,
                                           size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 512;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads4<128><<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(512, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<512><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<512><<<grid1, block1>>>(
            d_input + (blockNumber * 128 * 2 * 2), d_input, remain);
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

void testReduceSumUnrollAllIterationLoads8(float *input, float *result,
                                           size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 1024;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 2 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads8<128><<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(1024, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<1024><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<1024><<<grid1, block1>>>(
            d_input + (blockNumber * 128 * 2 * 2 * 2), d_input, remain);
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

void testReduceSumUnrollAllIterationLoads4_256Threads(float *input,
                                                      float *result,
                                                      size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 1024;
    internalSize = internalSize - remain;

    dim3 block(256, 1);
    size_t blockNumber = internalSize / 256 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads4<256><<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(512, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<512><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<512><<<grid1, block1>>>(
            d_input + (blockNumber * 256 * 2 * 2), d_input, remain);
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

void testReduceSumUnrollAllIterationLoads8LinearAccess(float *input,
                                                       float *result,
                                                       size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 1024;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 2 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads8LinearAccessMemory<128>
          <<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(1024, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<1024><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<1024><<<grid1, block1>>>(
            d_input + (blockNumber * 128 * 2 * 2 * 2), d_input, remain);
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

void testReduceSumUnrollAllIterationLoads8VectorLinearAccess(float *input,
                                                             float *result,
                                                             size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 1024;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 2 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads8VectorLinearAccessMemory<128>
          <<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(1024, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<1024><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<1024><<<grid1, block1>>>(
            d_input + (blockNumber * 128 * 2 * 2 * 2), d_input, remain);
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

void testReduceSumUnrollAllIterationLoads16LinearAccess(float *input,
                                                        float *result,
                                                        size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 2048;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128 / 16;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads16LinearAccessMemory<128>
          <<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(1024, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<1024><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<1024><<<grid1, block1>>>(d_input + (blockNumber * 128 * 16),
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

void testReduceSumUnrollAllIterationsLoads4_512Threads(float *input,
                                                       float *result,
                                                       size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 2048;
    internalSize = internalSize - remain;

    dim3 block(512, 1);
    size_t blockNumber = internalSize / 512 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads4<512><<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(1024, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<1024><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<1024><<<grid1, block1>>>(
            d_input + (blockNumber * 512 * 2 * 2), d_input, remain);
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

void testReduceSumUnrollAllIterationsLoads4_512ThreadsLinearAccess(
    float *input, float *result, size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 2048;
    internalSize = internalSize - remain;

    dim3 block(512, 1);
    size_t blockNumber = internalSize / 512 / 2 / 2;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumUnrollAndLoads4LinearAccess<512>
          <<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(1024, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<1024><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<1024><<<grid1, block1>>>(
            d_input + (blockNumber * 512 * 2 * 2), d_input, remain);
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

void testReduceSumByWrap(float *input, float *result, size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  size_t internalSize = size;
  while (1) {
    size_t remain = internalSize % 128;
    internalSize = internalSize - remain;

    dim3 block(128, 1);
    size_t blockNumber = internalSize / 128;
    dim3 grid(blockNumber, 1);

    if (blockNumber > 0) {
      reduceSumByWrap<<<grid, block>>>(d_input, d_input);
    }

    cudaDeviceSynchronize();

    if (remain > 0) {
      dim3 block1(128, 1);
      dim3 grid1(1, 1);

      if (blockNumber == 0) {
        reduceLast<128><<<grid1, block1>>>(d_input + 1, d_input, remain - 1);
      } else {
        reduceLast<128>
            <<<grid1, block1>>>(d_input + (blockNumber * 128), d_input, remain);
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

void testReduceSumByWrapV2(float *input, float *result, size_t size) {
  // device memory
  float *d_input;
  cudaMalloc(&d_input, sizeof(float) * size);
  cudaMemcpy(d_input, input, sizeof(float) * size, cudaMemcpyHostToDevice);

  int threads = 1024;
  int blocks = min((int)((size + threads - 1) / threads), (int)1024);

  reduceSumByWrapV2<<<blocks, threads>>>(d_input, d_input, size);
  reduceSumByWrapV2<<<1, 1024>>>(d_input, d_input, blocks);

  cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
}