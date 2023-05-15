#include <iostream>

#include "glog/logging.h"
#include "tars/core/buffer.h"
#include "tars/core/macro.h"
#include "tars/core/types.h"

int main() {
  // Elementwise Op test
  LOG(INFO) << "tars::core::Buffer test";

  ace::Buffer<float> buffer(ace::RuntimeType::CPU, int32, 1024);

  // LOG(INFO) << "capacity: " << buffer.capacity();

  return 0;
}