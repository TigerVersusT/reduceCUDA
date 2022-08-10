#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "legacy.h"
#include "reduceSum.h"

void serialCpu(float *input, float *result, const int size) {
  *result = 0;
  for (int i = 0; i < size; i++) {
    *result += input[i];
  }
}

int main() {
  const size_t size = 32 * 1024 * 1024;

  float result = 0;
  std::vector<float> h_input(size, 0);
  srand((unsigned)time(NULL));
  for (int i = 0; i < size; i++) {
    h_input[i] = 1;
  }

  // testReduceSum();
  // testReduceSumSharedMemory(h_input.data(), &result, size);
  // testReduceSumTwoLoads(h_input.data(), &result, size);
  // testReduceSumUnrollLastWrap(h_input.data(), &result, size);
  // testReduceSumUnrollAllWrap(h_input.data(), &result, size);
  // testReduceSumUnroll256(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.80
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      342,049
  //  Memory [%]            %                          82.50
  //  SOL DRAM              %                          82.50
  //  Duration              usecond                    250.30
  //  SOL L1/TEX Cache      %                          61.14
  //  SOL L2 Cache          %                          30.07
  //  SM Active Cycles      cycle                      337,261.22
  //  SM [%]                %                          54.30
  // testReduceSumUnroll128(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.80
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      348,754
  //  Memory [%]            %                          80.91
  //  SOL DRAM              %                          80.91
  //  Duration              usecond                    255.36
  //  SOL L1/TEX Cache      %                          51.04
  //  SOL L2 Cache          %                          29.49
  //  SM Active Cycles      cycle                      344,298.43
  //  SM [%]                %                          44.35
  // testReduceSumUnrollAlliterationsUseRegister(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.78
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      319030
  //  Memory [%]            %                          88.28
  //  SOL DRAM              %                          88.28
  //  Duration              usecond                    234.05
  //  SOL L1/TEX Cache      %                          32.74
  //  SOL L2 Cache          %                          31.75
  //  SM Active Cycles      cycle                      315,965.44
  //  SM [%]                %                          29.06
  // testReduceSumUnrollAllIterationLoads4(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      317,563
  //  Memory [%]            %                          88.62
  //  SOL DRAM              %                          88.62
  //  Duration              usecond                    232.74
  //  SOL L1/TEX Cache      %                          21.31
  //  SOL L2 Cache          %                          47.55
  //  SM Active Cycles      cycle                      314,594.09
  //  SM [%]                %                          19.46
  // testReduceSumUnrollAllIterationLoads8(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      317,536
  //  Memory [%]            %                          88.63
  //  SOL DRAM              %                          88.63
  //  Duration              usecond                    232.86
  //  SOL L1/TEX Cache      %                          35.61
  //  SOL L2 Cache          %                          31.68
  //  SM Active Cycles      cycle                      314,628.97
  //  SM [%]                %                          32.85
  // testReduceSumUnrollAllIterationLoads4_256Threads(h_input.data(), &result,
  //                                                size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      318,004
  //  Memory [%]            %                          88.48
  //  SOL DRAM              %                          88.48
  //  Duration              usecond                    232.12
  //  SOL L1/TEX Cache      %                          89.35
  //  SOL L2 Cache          %                          40.24
  //  SM Active Cycles      cycle                      314,906.72
  //  SM [%]                %                          19.14
  // testReduceSumUnrollAllIterationLoads8LinearAccess(h_input.data(), &result,
  //                                                  size);

  //  DRAM Frequency        cycle/nsecond              6.80
  //  SM Frequency          cycle/nsecond              1.37
  //  Elapsed Cycles        cycle                      523,348
  //  Memory [%]            %                          53.75
  //  SOL DRAM              %                          53.75
  //  Duration              usecond                    382.98
  //  SOL L1/TEX Cache      %                          97.92
  //  SOL L2 Cache          %                          19.16
  //  SM Active Cycles      cycle                      518,682.57
  //  SM [%]                %                          8.85
  // testReduceSumUnrollAllIterationLoads16LinearAccess(h_input.data(), &result,
  //                                                   size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      317,454
  //  Memory [%]            %                          88.62
  //  SOL DRAM              %                          88.62
  //  Duration              usecond                    232.51
  //  SOL L1/TEX Cache      %                          40.16
  //  SOL L2 Cache          %                          31.56
  //  SM Active Cycles      cycle                      314,049.96
  //  SM [%]                %                          38.32
  // testReduceSumUnrollAllIterationsLoads4_512Threads(h_input.data(), &result,
  //                                                  size);

  //  DRAM Frequency        cycle/nsecond              6.80
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      344.259
  //  Memory [%]            %                          81.71
  //  SOL DRAM              %                          81.71
  //  Duration              usecond                    252.13
  //  SOL L1/TEX Cache      %                          63.92
  //  SOL L2 Cache          %                          29.10
  //  SM Active Cycles      cycle                      341,230.43
  //  SM [%]                %                          35.32
  // testReduceSumUnrollAllIterationsLoads4_512ThreadsLinearAccess(h_input.data(),
  //                                                              &result,
  //                                                              size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      317,749
  //  Memory [%]            %                          88.56
  //  SOL DRAM              %                          88.56
  //  Duration              usecond                    232.50
  //  SOL L1/TEX Cache      %                          31.0
  //  SOL L2 Cache          %                          44.85
  //  SM Active Cycles      cycle                      314,426.82
  //  SM [%]                %                          12.15
  // testReduceSumUnrollAllIterationLoads8VectorLinearAccess(h_input.data(),
  //                                                        &result, size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      2,203,537
  //  Memory [%]            %                          14.70
  //  SOL DRAM              %                          0.00
  //  Duration              usecond                    1610
  //  SOL L1/TEX Cache      %                          15.76
  //  SOL L2 Cache          %                          5.09
  //  SM Active Cycles      cycle                      2,201,373.46
  //  SM [%]                %                          14.70
  // testReduceSumByWrap(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      701,179
  //  Memory [%]            %                          50.18
  //  SOL DRAM              %                          40.18
  //  Duration              usecond                    513.63
  //  SOL L1/TEX Cache      %                          51.94
  //  SOL L2 Cache          %                          14.69
  //  SM Active Cycles      cycle                      696,767.04
  //  SM [%]                %                          50.08
  // testReduceSumUnrollAllIterationsWrapReduce(h_input.data(), &result, size);

  //  DRAM Frequency        cycle/nsecond              6.79
  //  SM Frequency          cycle/nsecond              1.36
  //  Elapsed Cycles        cycle                      318,146
  //  Memory [%]            %                          88.46
  //  SOL DRAM              %                          88.46
  //  Duration              usecond                    233.12
  //  SOL L1/TEX Cache      %                          89.30
  //  SOL L2 Cache          %                          40.09
  //  SM Active Cycles      cycle                      315,140.16
  //  SM [%]                %                          19.43
  testReduceSumByWrapV2(h_input.data(), &result, size);

  std::cout << "gpu result: " << result << std::endl;
  return 0;
}
