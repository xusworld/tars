#include "tars/core/tensor.h"

namespace tars {

std::string Tensor::DebugString() { return ""; }

// returns batch size of the tensor
int32_t Tensor::batch() const { return shape_[0]; }

// returns channel of the tensor
int32_t Tensor::channel() const {
  if (dformat_ == DataFormat_NCHW) {
    return shape_[1];
  } else if (dformat_ == DataFormat_NHWC) {
    return shape_[3];
  }
  return -1;
}

// returns height of the tensor
int32_t Tensor::height() const {
  if (dformat_ == DataFormat_NCHW) {
    return shape_[2];
  } else if (dformat_ == DataFormat_NHWC) {
    return shape_[1];
  }
  return -1;
}

// returns width of the tensor
int32_t Tensor::width() const {
  if (dformat_ == DataFormat_NCHW) {
    return shape_[3];
  } else if (dformat_ == DataFormat_NHWC) {
    return shape_[2];
  }
  return -1;
}

}  // namespace tars