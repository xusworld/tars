#ifndef TARS_CORE_SHAPE_TILE_H_
#define TARS_CORE_SHAPE_TILE_H_

#include <vector>

#include "tars/core/macro.h"
#include "tars/core/shape/shape_infer.h"

namespace tars {
namespace core {

/* Constructs a tensor by tiling a given tensor.
 *
 * This operation creates a new tensor by replicating input multiples times. The
 * output tensor's i'th dimension has input.dims(i) * multiples[i] elements, and
 * the values of input are replicated multiples[i] times along the 'i'th
 * dimension.
 *
 * For example, tiling [a b c d] by [2] produces [a b c d a b c d].
 */
class TileShapeInfer : public ShapeInfer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(TileShapeInfer);

  virtual ~TileShapeInfer() = default;

  // inputs shape --> outputs shape
  virtual Status run(const tars::Op* op, const std::vector<Tensor*>& inputs,
                     std::vector<Tensor*>& outputs) {
    DLOG(INFO) << "Tile op shape inference ...";
    // check inputs and outputs number
    CHECK(inputs.size() == 2) << ", tile ops should have two inputs.";
    CHECK(outputs.size() == 1) << ", tile ops shoule have one output.";

    auto input = inputs[0];
    auto multiples = inputs[1];

    // check inputs' attributes
    CHECK(input->rank() == multiples->size())
        << ", input's rank should equal to multiples' element size";
    CHECK(multiples->dtype() == DataType_DT_INT32)
        << ", multiples tensor should be a int32 tensor.";
    CHECK(multiples->rank() == 1)
        << ", multiples tensor should be a 1-D tensor.";

    // inference the shape of output tensor
    TensorShape shape;
    for (int i = 0; i < multiples->size(); ++i) {
      shape[i] = input->shape()[i] * multiples->data<int32_t>()[i];
    }

    // set output's tensor shape
    outputs[0]->reshape(shape);
    // set output's data type
    outputs[0]->astype(input->dtype());
    // set output's data format
    outputs[0]->set_dformat(inputs[0]->dformat());

    // debug log
    std::vector<int32_t> multiples_vec;
    for (int i = 0; i < multiples->size(); ++i) {
      multiples_vec.push_back(multiples->data<int32_t>()[i]);
    }

    DLOG(INFO) << "input's shape: " << input->shape();
    DLOG(INFO) << "multiples's content: " << multiples_vec;
    DLOG(INFO) << "output's shape: " << outputs[0]->shape();

    return Status::OK();
  }
};

}  // namespace core
}  // namespace tars

#endif  // TARS_CORE_SHAPE_TILE_H_