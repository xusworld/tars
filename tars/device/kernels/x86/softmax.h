#pragma once

#include <iostream>
#include <vector>

namespace ace {
namespace device {
namespace x86 {

void _avx2_softmax(float* outputs, const float* inputs, const int32_t size);

}
}  // namespace device
}  // namespace ace
