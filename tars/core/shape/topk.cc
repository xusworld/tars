#ifndef TARS_CORE_SHAPE_TOPK_H_
#define TARS_CORE_SHAPE_TOPK_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Finds values and indices of the k largest elements for the last dimension.
class TopKShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(TopKShapeInfer);

  virtual ~TopKShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "TopK op shape inference ...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 2 || inputs.size() == 3)
        << ", TopK ops should have two or three inputs.";
    CHECK(outputs.size() == 2) << ", concat ops shoule have two outputs.";

    auto input = inputs[0];
    auto k = inputs[1];

    CHECK(k->size() == 1) << ", k should be a scalar.";
    CHECK(k->dtype() == DataType_DT_INT32) << ", k should be a int32 tensor";

    // the value of scalar k
    const int ks = k->data<int32_t>()[0];

    // the value of axis
    int axis = (inputs.size() == 3 ? inputs[2]->data<int32_t>()[0]
                                   : input->rank() - 1);
    if (axis < 0) axis += input->rank();

    // Returns:
    //   - Output values : The k largest elements along each last dimensional
    //   slice
    //   - Output indices : The indices of values within the last dimension of
    //   input

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

#endif  // TARS_CORE_SHAPE_TOPK_H_