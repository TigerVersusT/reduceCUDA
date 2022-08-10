#include "cuda_runtime.h"

extern void testReduceSumUnrollAlliterationsUseRegister(float *input,
                                                        float *result,
                                                        size_t size);

extern void testReduceSumUnrollAllIterationLoads4(float *input, float *result,
                                                  size_t size);

extern void testReduceSumUnrollAllIterationLoads8(float *input, float *result,
                                                  size_t size);

extern void testReduceSumUnrollAllIterationLoads4_256Threads(float *input,
                                                             float *result,
                                                             size_t size);

extern void testReduceSumUnrollAllIterationLoads8LinearAccess(float *input,
                                                              float *result,
                                                              size_t size);

extern void testReduceSumUnrollAllIterationLoads16LinearAccess(float *input,
                                                               float *result,
                                                               size_t size);

extern void testReduceSumUnrollAllIterationsLoads4_512Threads(float *input,
                                                              float *result,
                                                              size_t size);

extern void testReduceSumUnrollAllIterationsLoads4_512ThreadsLinearAccess(
    float *input, float *result, size_t size);

extern void testReduceSumUnrollAllIterationLoads8VectorLinearAccess(
    float *input, float *result, size_t size);

extern void testReduceSumByWrap(float *input, float *result, size_t size);

extern void testReduceSumUnrollAllIterationsWrapReduce(float *input,
                                                       float *result,
                                                       size_t size);

extern void testReduceSumByWrapV2(float *input, float *result, size_t size);