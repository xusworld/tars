#pragma once

#include "tars/core/macro.h"
#include "tars/ir/op_generated.h"
#include "tars/ir/op_option_generated.h"

namespace tars {
namespace device {

struct OpContext {
  std::string name;
  OpOptionUnion option;
  DataType dtype;
};

}  // namespace device
}  // namespace tars