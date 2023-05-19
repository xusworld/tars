#pragma once

#include <iostream>
#include <vector>

namespace ace {
namespace device {
namespace x86 {

float _avx2_dot_product(const float* x, const float* y, int n);

}  // namespace x86
}  // namespace device
}  // namespace ace