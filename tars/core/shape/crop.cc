#ifndef TARS_CORE_SHAPE_CROP_H_
#define TARS_CORE_SHAPE_CROP_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Concatenates tensors along one dimension.
class CropShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(CropShapeInfer);

  virtual ~CropShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "CastShapeInfer...";
    // check inputs number and outputs number
    CHECK(inputs.size() >= 2)
        << ", concat ops should have more than two inputs.";
    CHECK(outputs.size() == 1) << ", concat ops shoule have one output.";

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

#endif  // TARS_CORE_SHAPE_CROP_H_