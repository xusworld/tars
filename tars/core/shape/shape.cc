#ifndef TARS_CORE_SHAPE_SHAPE_H_
#define TARS_CORE_SHAPE_SHAPE_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Returns a tensor containing the shape of the input tensor.returns a 1-D
// integer tensor representing the shape of input. For a scalar input, the
// tensor returned has a shape of (0,) and its value is the empty vector
class ShapeShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(ShapeShapeInfer);

  virtual ~ShapeShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "Shape op shape inference ...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 1) << ", shape ops should have one input.";
    CHECK(outputs.size() == 1) << ", concat ops shoule have one output.";

    // set output's tensor shape
    outputs[0]->reshape({1});
    // set output's data type
    outputs[0]->astype(DataType_DT_INT32);
    // set output's data format
    outputs[0]->set_dformat(inputs[0]->dformat());

    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_SHAPE_H_