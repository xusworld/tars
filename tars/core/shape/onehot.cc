#ifndef TARS_CORE_SHAPE_ONEHOT_H_
#define TARS_CORE_SHAPE_ONEHOT_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

/**
 * Returns a one-hot tensor.
 *
 * one_hot(indices, depth, on_value=None, off_value=None, axis=None)
 *
 The locations represented by indices in indices take value on_value, while
 * all other locations take value off_value.
 * on_value and off_value must have matching data types. If dtype is also
 * provided,  they must be the same data type as specified by dtype.
 * If on_value is not provided, it will default to the value 1 with type dtype
 * If off_value is not provided, it will default to the value 0 with type dtype
 * If the input indices is rank N, the output will have rank N+1. The new axis
 * is created at dimension axis (default: the new axis is appended at the end).
 */
class OneHotShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(OneHotShapeInfer);

  virtual ~OneHotShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "OneHot op shape inference ...";

    // check inputs and outputs number
    CHECK(inputs.size() == 4) << ", onehot ops should have four input.";
    CHECK(outputs.size() == 1) << ", onehot ops shoule have one output.";

    auto indices = inputs[0];
    auto depth = inputs[1];
    // A scalar defining the depth of the one hot dimension.
    const int depth = depth->data<int32_t>()[0];
    if (depth < 0) {
      return Status::ERROR();
    }

    // set output's tensor shape
    outputs[0]->reshape({1});
    // set output's data type
    outputs[0]->astype(DataType_DT_INT32);
    // set output's data format
    outputs[0]->set_dformat(inputs[0]->dformat());

    return Status::OK();

    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_ONEHOT_H_