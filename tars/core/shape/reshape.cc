#ifndef TARS_CORE_SHAPE_RESHAPE_H_
#define TARS_CORE_SHAPE_RESHAPE_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Given tensor, this operation returns a new tf.Tensor that has the same values
// as tensor in the same order, except with a new shape given by shape.
class ReshapeShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(ReshapeShapeInfer);

  virtual ~ReshapeShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "Reshape op shape inference ...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 1 || inputs.size() == 2)
        << ", reshape ops should have one or two inputs.";
    CHECK(outputs.size() == 1) << ", reshape ops shoule have one output.";

    // set output's tensor shape
    outputs[0]->reshape(inputs[0]->shape());
    // set output's data type
    outputs[0]->astype(inputs[0]->dtype());
    // set output's data format
    outputs[0]->set_dformat(inputs[0]->dformat());

    DLOG(INFO) << "input shape: " << inputs[0]->shape();
    DLOG(INFO) << "output shape: " << outputs[0]->shape();

    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_RESHAPE_H_