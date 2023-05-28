#include "utils.h"

namespace tars {
namespace device {
namespace cuda {

template <typename InputDtype, typename OutputDtype>
__global__ void _cuda_cast(OutputDtype* out_data, const InputDtype* in_data,
                           const int num_threads) {
  CUDA_KERNEL_LOOP(tid, num_threads) {
    out_data[tid] = static_cast<OutputDtype>(in_data[tid]);
  }
}

}  // namespace cuda
}  // namespace device
}  // namespace tars