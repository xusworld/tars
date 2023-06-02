#ifndef TARS_CORE_SHAPE_TRANSPOSE_H_
#define TARS_CORE_SHAPE_TRANSPOSE_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Concatenates tensors along one dimension.
class TransposeShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(TransposeShapeInfer);

  virtual ~TransposeShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "Transpose op shape inference ...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 2) << ", transpose ops should have two inputs.";
    CHECK(outputs.size() == 1) << ", transpose ops shoule have one output.";

    auto input = inputs[0];
    // Permutes the dimensions according to the value of perm.
    auto perm = inputs[1];

    const int dims = input->buffer().dimensions;
    if (perm->getType().code != halide_type_int || 32 != perm->getType().bits ||
        dims != perm->buffer().dim[0].extent) {
      return false;
    }

    // set output's data format
    outputs[0]->set_dformat(inputs[0]->dformat());
    // set output's tensor shape
    outputs[0]->reshape(inputs[0]->shape());
    // set output's data type
    const auto opParam = op->main_as_CastParam();
    outputs[0]->set_dtype(opParam->dstT());

    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_TRANSPOSE_H_