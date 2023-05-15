#pragma once

#include "tars/ir/op_generated.h"
#include "tars/ir/op_option_generated.h"

namespace ace {
namespace device {

// OpParam 中持有 operator 的配置信息
struct OpParam {
  OpOptionUnion option;
  DataType dtype;
};

}  // namespace device
}  // namespace ace