#ifndef TARS_CORE_SHAPE_UNIQUE_H_
#define TARS_CORE_SHAPE_UNIQUE_H_

#include <unordered_set>
#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

/* Finds unique elements in a 1-D tensor.
 *
 * x = [4, 5, 1, 2, 3, 3, 4, 5]
 * y, idx = unique(x)
 *   y ==>  [ 4, 5, 1, 2, 3 ]
 *   idx ==> [ 0, 1, 2, 3, 4, 4, 0, 1 ]
 */
class UniqueShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(UniqueShapeInfer);

  virtual ~UniqueShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "UniqueShapeInfer ...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 1) << ", unique op should have one input.";

    CHECK(outputs.size() == 1 || outputs.size() == 2)
        << ", unique op should have one output or two outputs.";

    // check input is a 1-D tensor
    CHECK(inputs[0]->rank() == 1)
        << ", unique op's input should be a 1-D tensor.";

    std::unordered_set<int> unique_values;
    for (int i = 0; i < inputs[0]->size(); ++i) {
      unique_values.insert(inputs[0]->data<int>()[i]);
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

#endif  // TARS_CORE_SHAPE_UNIQUE_H_