#ifndef TARS_CORE_SHAPE_ARGMAX_H_
#define TARS_CORE_SHAPE_ARGMAX_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

// Returns the index with the largest value across dimensions of a tensor.
class ArgmaxShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(ArgmaxShapeInfer);

  virtual ~ArgmaxShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "ArgMaxShapeInfer...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 1) << ", binary ops should have one input.";
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

#endif  // TARS_CORE_SHAPE_ARGMAX_H_