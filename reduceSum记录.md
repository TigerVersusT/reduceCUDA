# reduceSum 算子实现版本记录总结

## 第一版

~~~C++
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
~~~
主要思想是在一个block内每次规约启动左边一半的线程，每一个线程完成当前规模的一次原地规约，随后规模减半，直到为一，此时
该块的规约结果就放在线程0管理的地址处。

这个kernel没有线程束分化的问题，因为每次激活一半的线程，同一个线程束内的线程要么同时启动，要么不参与运算，不存在一部分
线程选择A分支，另一部分线程选择B分支的情况。

## 第二版

~~~C++
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
~~~
主要思想是由于全局内存的存取效率相对低于对共享内存的存取，所以第一步可以先将数据从全局内存读取到共享内存中，随后在共享内存中
进行规约。

主要结果如下：
| dataseize        |         | 10*3*1024*1024 |
| :--------------- | :------ | :------------- |
| Duration         | usecond | 744.58         |
| SOL DRAM         | %       | 26.22          |
| SOL L1/TEX Cache | %       | 94.54          |
| SOL L2 Cache     | %       | 9.81           |
| SM [%]           | %       | 75.04          |

相比于第一版并没有时间上的进步，但是提高了L1 Cache的利用率，主要原因在于算子的计算密度并不高，主要限制在于内存带宽的利用率。虽然
在共享内存中规约肯定会比从全局内存中规约要快，但是算子把数据从全局内存读取到共享内存这一步效率并不高，即使能够很快的完成共享内存
规约，也会因为等待数据时间过长导致没有任何进步。

## 第三版

~~~C++
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
~~~
这个kernel的主要改进点在于，kernel执行的时候每次规约都用一半的线程不活跃，造成线程资源的浪费，因此，在上一步的基础上每次读取两个数据，并进行规约
结果如下：
| dataseize        |         | 10*3*1024*1024 |
| :--------------- | :------ | :------------- |
| Duration         | usecond | 372.22         |
| SOL DRAM         | %       | 52.08          |
| SOL L1/TEX Cache | %       | 91.3           |
| SOL L2 Cache     | %       | 19.09          |
| SM [%]           | %       | 73.88          |

相比于上一个kernel，这次的改进导致时间接近减半，内存利用率翻倍。最初改进的理由是避免线程的浪费，从SM利用率上看的确实现了这一目标，但是kernel的性能
改进主要得益于内存带宽的利用率提高。同一个线程每次读取的数据增加了，全局层面上会有更多的访存被合并，提高了数据读取到共享内存的效率，减少了数据等待的时间
自然提高了性能。

## 第四版

~~~C++
__device__ void wrapReduce(volatile float *input, int id) {
  input[id] += input[id + 32];
  input[id] += input[id + 16];
  input[id] += input[id + 8];
  input[id] += input[id + 4];
  input[id] += input[id + 2];
  input[id] += input[id + 1];
}
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
~~~
这一版的改进出发点在于，当规约规模小于32时只需要一个线程束就能实现所有的功能，而GPU上线程的基本调度单位是线程束，在同一个线程束内
的所有线程使用SIMD的方式同步执行。因此，当规模小于等于32时对循环进行展开，直接在一个束内进行求解，这样的好处是避免了所有线程在规模
小于等于32时执行无效的for循环和if判断指令，提高了线程的利用效率。

结果如下：
| dataseize        |         | 10*3*1024*1024 |
| :--------------- | :------ | :------------- |
| Duration         | usecond | 247.52         |
| SOL DRAM         | %       | 78.40          |
| SOL L1/TEX Cache | %       | 59.03          |
| SOL L2 Cache     | %       | 28.79          |
| SM [%]           | %       | 52.44          |

从结果看，随着内存带宽利用率的提高，算子耗时进一步减少，主要原因在于通过展开最后一个线程束，减少了线程的无效指令执行时间，从而间接
增加了线程访存的时间。

## 第五版

~~~C++
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
  if (threadIdx.x < 32) wrapReduce<128>(tempArray, threadIdx.x);
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
~~~

这一版的改进思路延续上一版的循环展开，既然最后一个束展开后可以减少for循环和if判断指令的消耗，那么在明确线程块大小的前提下也可以通过
循环展开减少循环指令和if指令的消耗。

结果如下：
| dataseize        |         | 10*3*1024*1024 |
| :--------------- | :------ | :------------- |
| Duration         | usecond | 239.42         |
| SOL DRAM         | %       | 80.80          |
| SOL L1/TEX Cache | %       | 60.81          |
| SOL L2 Cache     | %       | 29.63          |
| SM [%]           | %       | 54.01          |

和上一版只展开最后一个束相比，推理时间进一步减少，这种循环展开带来的性能提升主要是减少了不必要的指令执行，增加了访存指令的密度，间接
提高了访存时间和内存带宽利用率。

## 第六版

~~~C++
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
~~~
这个kernel的优化思路延续第三版，既然同一个线程读取两个数据会提高内存带宽的利用率，那么增加更多的数据读取应该能进一步增带宽利用率，
这一版是一个线程块处理连续的四个数据块。

结果如下：
| dataseize        |               | 32*1024*1024 |
| :--------------- | :------------ | :----------- |
| DRAM Frequency   | cycle/nsecond | 6.78         |
| SM Frequency     | cycle/nsecond | 1.36         |
| Elapsed Cycles   | cycle         | 319030       |
| Memory [%]       | %             | 88.28        |
| SOL DRAM         | %             | 88.28        |
| Duration         | usecond       | 234.05       |
| SOL L1/TEX Cache | %             | 32.74        |
| SOL L2 Cache     | %             | 31.75        |
| SM Active Cycles | cycle         | 315,965.44   |
| SM [%]           | %             | 29.06        |

这一版的内存利用率达到了88.28，推理时间相比于之前的版本也有了更多的进步，提升的主要原因就在于进一步地增加了可供调度的访存指令的数量，
增加了访存指令的密度，从而提高内存带宽的利用率。

# 第七版

~~~C++
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
~~~

这个kernel的优化思路延续第六版，既然增加独立的访存指令能够提高内存带宽利用率，那么就可以继续增加，这个版本一个块需要处理原来8倍的block。

结果如下：
| dataseize        |               | 32*1024*1024 |
| :--------------- | :------------ | :----------- |
| DRAM Frequency   | cycle/nsecond | 6.79         |
| SM Frequency     | cycle/nsecond | 1.36         |
| Elapsed Cycles   | cycle         | 317,563      |
| Memory [%]       | %             | 88.62        |
| SOL DRAM         | %             | 88.62        |
| Duration         | usecond       | 232.74       |
| SOL L1/TEX Cache | %             | 21.31        |
| SOL L2 Cache     | %             | 47.55        |
| SM Active Cycles | cycle         | 314,594.09   |
| SM [%]           | %             | 19.46        |

一次读取8个数据并进行规约进一步的提高了内存带宽的利用率，从而减少了推理时间，由于一个线程需要处理原来八倍的数据量，所需的block数量大大降低，
所以SM的利用率反而降低了

## 第八版

~~~C++
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
~~~

这一版尝试继续增加单个线程处理的数据量。

结果如下：
| dataseize        |               | 32*1024*1024 |
| :--------------- | :------------ | :----------- |
| DRAM Frequency   | cycle/nsecond | 6.80         |
| SM Frequency     | cycle/nsecond | 1.37         |
| Elapsed Cycles   | cycle         | 523,348      |
| Memory [%]       | %             | 53.75        |
| SOL DRAM         | %             | 53.75        |
| Duration         | usecond       | 382.98       |
| SOL L1/TEX Cache | %             | 97.92        |
| SOL L2 Cache     | %             | 19.16        |
| SM Active Cycles | cycle         | 518,682.57   |
| SM [%]           | %             | 8.85         |

从结果看并没有达到理想的效果，一方面是因为这种通过增加单个线程负载的方式来提高内存带宽利用率已经接近极限了，在上一个版本中的
内存利用率提升已经很小了再增加负载效果有限。

性能衰退的原因可能是因为寄存器资源不足，导致单个线程不能按照设定的逻辑将数据存放
到寄存器中然后直接规约。实际上这一系列优化的主要思路就是通过给单个线程分配更多的寄存器资源和增加更多的内存访问实现对内存带宽
的利用率增加，这一方法的一个限制就在于SM上的寄存器资源，当资源不足时，寄存器就无法放置这么多全局内存数据，转而将数据存放在全局内存
导致性能下降。

# 第九版

~~~C++
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
~~~

这一版在第七版的基础上进行优化，主要考虑是第七版读取八个全局内存数据的时候间隔很大，这样可能会造成分散的内存访问，降低内存带宽利用率。

结果如下：
| dataseize        |               | 32*1024*1024 |
| :--------------- | :------------ | :----------- |
| DRAM Frequency   | cycle/nsecond | 6.79         |
| SM Frequency     | cycle/nsecond | 1.36         |
| Elapsed Cycles   | cycle         | 318,004      |
| Memory [%]       | %             | 88.48        |
| SOL DRAM         | %             | 88.48        |
| Duration         | usecond       | 232.12       |
| SOL L1/TEX Cache | %             | 89.35        |
| SOL L2 Cache     | %             | 40.24        |
| SM Active Cycles | cycle         | 314,906.72   |
| SM [%]           | %             | 19.14        |

和第七版相比，推理时间有一点进步，但是并不是和预想的那样的进步，主要原因在于，块内的合并内存访问极大的提高了L1 Cache的利用率，所以减少了
规约操作的时间，但是块内的合并内存访问并没有提高全局的内存合并访问，从结果上看甚至降低了内存利用率，全靠L1 Cache的提高平衡了一些性能。

## 第十版

~~~C++
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
~~~

这一版优化的出发点是，在展开最后一个束的时候，第四版是将数据从共享内存读入寄存器，规约完成后再写回全局内存，这就存在重复的数据读写和运算指令，
通过一次读数据后直接使用shuffle指令即可实现线程束内部的规约，减少重复的数据读写和运算指令消耗。

结果差不多，几乎没有区别。

分析原因，访问寄存器只需要一个周期，访问共享内存是1~20个周期，理论上寄存器应该更快，但是算子的主要瓶颈仍然在于全局内存访问效率，即使规约操作
变快了还是需要等待数据，因此这样的优化意义不大。另一方面，由于对共享内存的访问不存在bank conflict 的问题，本身访问共享内存的带宽就足够大，和
访问寄存器差异不大。

## 第十一版

~~~C++
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
~~~

这一版的优化思路建立在对之前的kernel的数据流动分析基础上，以目前最好的第九版为例：数据首先从全局内存流向寄存器，然后从寄存器流向共享内存，这就是
第一步的数据导入；在接下来的基于共享内存的规约中，每进行一次规约活跃线程数量减半，造成线程资源的浪费。

为了解决这两个问题，可以使用基于束的规约方式：第一步中，每个线程束获取到全局内存数据后直接使用shuffle指令进行束内规约，然后将结果写入共享内存，这样即可
解决第一个问题；在基于共享内存的规约过程中，每个线程束处理32个数据，读入数据后进行束内规约，然后将结果写入共享内存。使用这样的规约方式既能够优化数据
流动，又可以优化线程资源使用。

结果如下：
| dataseize        |               | 32*1024*1024 |
| :--------------- | :------------ | :----------- |
| DRAM Frequency   | cycle/nsecond | 6.79         |
| SM Frequency     | cycle/nsecond | 1.36         |
| Elapsed Cycles   | cycle         | 2,203,537    |
| Memory [%]       | %             | 14.70        |
| SOL DRAM         | %             | 0.00         |
| Duration         | usecond       | 1600         |
| SOL L1/TEX Cache | %             | 15.76        |
| SOL L2 Cache     | %             | 5.09         |
| SM Active Cycles | cycle         | 2,201,373.46 |
| SM [%]           | %             | 14.70        |

结果看来，这个kernel性能非常差，并没有达到预期效果。主要原因在于，当线程束读取全局内存数据的时候采用的是束内连续读取，这样的方式在微观上能够一次读取32个
数据，但是从全局上并不能导致合并的内存读取，因此内存的利用率十分低下。

## 第十二版

~~~C++
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
~~~

这版本参考了Nvidia 的博客，在上一个版本的基础上，使用grid-stride的方式读取全局内存而不是基于束的全局内存读取，其余操作不变。

结果如下：
| dataseize        |               | 32*1024*1024 |
| :--------------- | :------------ | :----------- |
| DRAM Frequency   | cycle/nsecond | 6.79         |
| SM Frequency     | cycle/nsecond | 1.36         |
| Elapsed Cycles   | cycle         | 318,146      |
| Memory [%]       | %             | 88.46        |
| SOL DRAM         | %             | 88.46        |
| Duration         | usecond       | 232.12       |
| SOL L1/TEX Cache | %             | 59.30        |
| SOL L2 Cache     | %             | 40.09        |
| SM Active Cycles | cycle         | 315,140.16   |
| SM [%]           | %             | 19.43        |

从结果上看，使用了grid-stride方式读取全局内存后，改进的wrap-based kernel性能能够达到之前通过增加内存事务一样的效果。这个kernel的好处在于，
使用grid-stride方式读取全局内存使得kernel具有可扩展性，之前的方式需要指定thread的数量刚好匹配的处理的数据的数量，如果要处理多个数据就需要
改变kernel调用的方式。使用新的wrap-based, grid-stride读取方式的kernel能够使用相同的代码调用，不需要复杂的调用函数。

## 还没有想明白的问题
完全循环展开为什么128效果最好，256，512效果不好，

## 如何设定block和grid
block中线程的数量应该大于一个SM中最大线程数/最大block数，并且选择SM中最大线程数的约数。由于block是在SM上运行，grid中block的数量最好是SM个数
的倍数。

## 其它收获
在优化reduce算子的过程中纠正了一些错误的认知，比如”使用共享内存一定会更快“，”一个block内的线程访问连续的地址会提高全局内存访问效率“，更重要的
是在分析算子的时候能够从数据流动的方式分析算子的性能，通过分析数据的流动分析算子是否有冗余的数据搬运或者逻辑分支。