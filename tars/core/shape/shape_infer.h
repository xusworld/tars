#ifndef TARS_CORE_SHAPE_SHAPE_INFER_H_
#define TARS_CORE_SHAPE_SHAPE_INFER_H_

#include <vector>

#include "glog/logging.h"
#include "ir/current/model_generated.h"
#include "tars/core/status.h"
#include "tars/core/tensor.h"

namespace tars {
namespace core {

// Op shape inference.
class ShapeInfer {
 public:
  virtual ~ShapeInfer() = default;

  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) const = 0;
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_SHAPE_INFER_H_