#ifndef TARS_CORE_SHAPE_BINARY_H_
#define TARS_CORE_SHAPE_BINARY_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Op shape inference.
class BinaryShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(BinaryShapeInfer);

  virtual ~BinaryShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "BinaryShapeInfer...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 2) << ", binary ops should have two inputs.";
    CHECK(outputs.size() == 1) << ", binary ops shoule have one output.";
    // check inputs data format
    CHECK(inputs[0]->dformat() == inputs[1]->dformat());
    // set outputs data format
    outputs[0]->set_dformat(inputs[0]->dformat());

    // broadcast if necessary
    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_BINARY_H_