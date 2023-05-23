#include <stdio.h>
#include <functional>
#include <cuda_runtime.h>

namespace kernels {
namespace cuda {

enum class ReduceDataType {
    REDUCE_INT32,
    REDUCE_FLOAT,
    REDUCE_DOUBLE,
};

enum class ReduceType {
    REDUCE_SUM,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
};

void ReduceBySinglePass(const float* input, float* output, size_t n);

void KernelExecTimecost(const std::string& title, std::function<void(void)> func);

} // namespace cuda
} // namespace kernel
