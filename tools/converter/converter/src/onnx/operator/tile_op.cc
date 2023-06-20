#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(TileOnnx);

ace::OpType TileOnnx::opType() { return ace::OpType_Tile; }

ace::OpParameter TileOnnx::type() { return ace::OpParameter_NONE; }

void TileOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                   OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(TileOnnx, Tile);

}  // namespace converter
}  // namespace ace