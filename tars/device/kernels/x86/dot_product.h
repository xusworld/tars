#pragma once

#include <iostream>
#include <vector>

namespace tars {
namespace device {
namespace x86 {

float _avx2_dot_product(const float* x, const float* y, int n);

}  // namespace x86
}  // namespace device
}  // namespace tars