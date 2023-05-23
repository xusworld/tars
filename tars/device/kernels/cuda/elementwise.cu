
#include "utils.h"

namespace ace {
namespace device {
namespace cuda {

template<typename Dtype>
__global__ void ker_relu_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count, Dtype neg_slop,
                             int in_n, int in_c, int in_h, int in_w,
                             int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                             int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {
    CUDA_KERNEL_LOOP(tid, count) {
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride
                     + c * in_c_stride
                     + h * in_h_stride
                     + w * in_w_stride;

        int out_idx =  n * out_n_stride
                       + c * out_c_stride
                       + h * out_h_stride
                       + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        out_data[out_idx] = in_var > Dtype(0) ? in_var : in_var * neg_slop;
    }
}

}
}
}