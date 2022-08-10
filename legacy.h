#include "cuda_runtime.h"

extern __global__ void reduceSum(float *input, float *result, const int size);
extern __global__ void reduceSumSharedMemory(float *input, float *result,
                                             const int size);
extern __global__ void reduceSumTwoLoads(float *input, float *result,
                                         const int size);

extern __global__ void reduceSumUnrollLastWrap(float *input, float *result,
                                               int size);
extern __global__ void reduceSumUnroll512(float *input, float *result,
                                          int size);

extern __global__ void reduceSumUnroll256(float *input, float *result,
                                          int size);
extern __global__ void reduceSumUnroll128(float *input, float *result);

extern __global__ void reduceSumLast256(float *input, float *result, int size);

extern void testReduceSum();
extern void testReduceSumSharedMemory(float *input, float *result, int size);
extern void testReduceSumUnroll128(float *input, float *result, size_t size);
extern void testReduceSumUnroll256(float *input, float *result, int size);
extern void testReduceSumUnroll512(float *input, float *result, int size);
extern void testReduceSumUnrollLastWrap(float *input, float *result, int size);
extern void testReduceSumTwoLoads(float *input, float *result, int size);