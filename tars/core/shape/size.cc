#ifndef TARS_CORE_SHAPE_SIZE_H_
#define TARS_CORE_SHAPE_SIZE_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Returns the size of a tensor. Returns a 0-D Tensor representing the number of
// elements in input of type out_type.

class SizeShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(SizeShapeInfer);

  virtual ~SizeShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "Size op shape inference ...";

    // check inputs and outputs number
    CHECK(inputs.size() == 1) << ", size ops should have one input.";
    CHECK(outputs.size() == 1) << ", size ops shoule have one output.";

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

#endif  // TARS_CORE_SHAPE_SIZE_H_