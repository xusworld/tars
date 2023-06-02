#ifndef TARS_CORE_SHAPE_SQUEEZE_H_
#define TARS_CORE_SHAPE_SQUEEZE_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

/* Removes dimensions of size 1 from the shape of a tensor.
 *
 * Given a tensor input, this operation returns a tensor of the same type with
 * all dimensions of size 1 removed. If you don't want to remove all size 1
 * dimensions, you can remove specific size 1 dimensions by specifying axis.
 *
 */
class SqueezeShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(SqueezeShapeInfer);

  virtual ~SqueezeShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "Squeeze op shape inference ...";
    // check inputs number and outputs number
    CHECK(inputs.size() == 1 || inputs.size() == 2)
        << ", squeeze ops should have one or two inputs.";
    CHECK(outputs.size() == 1) << ", squeeze ops shoule have one output.";

    // An optional list of ints. Defaults to []. If specified, only squeezes
    // the dimensions listed. The dimension index starts at 0. It is an error to
    // squeeze a dimension that is not 1.
    std::vector<int32_t> axis;

    if (inputs.size() == 2) {  // axis from another input tensor
      for (int i = 0; i < inputs[1]->size(); ++i) {
        axis.push_back(inputs[1]->data<int32_t>()[i]);
      }
    } else {  // axis from options
      auto data = op->main_as_SqueezeParam()->squeezeDims()->data();
      if (data != nullptr) {
        auto size = op->main_as_SqueezeParam()->squeezeDims()->size();
        for (int i = 0; i < size; ++i) {
          axis.push_back(data[i]);
        }
      }
    }

    // inference the shape of output tensor
    TensorShape shape;
    if (axis.size() != 0) {
      for (int i = 0; i < axis.size(); ++i) {
        auto dv = axis[i];
        if (dv < 0) dv += inputs[0]->rank();

        if (inputs[0]->shape()[axis[i]] != 1) {
          LOG(FATAL) << "input tensor's dim " << axis[i] << " shouble be 1";
        } else {
          shape.push_back(inputs[0]->shape()[axis[i]]);
        }
      }
    } else {
      for (int i = 0; i < inputs[0]->shape().size(); ++i) {
        if (inputs[0]->shape()[i] != 1) {
          shape.push_back(inputs[0]->shape()[i]);
        }
      }
    }

    // set output's tensor shape
    outputs[0]->reshape(shape);
    // set output's data type
    outputs[0]->astype(inputs[0]->dtype());
    // set output's data format
    outputs[0]->set_dformat(inputs[0]->dformat());

    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_SQUEEZE_H_