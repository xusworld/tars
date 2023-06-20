#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(RangeOnnx);

ace::OpType RangeOnnx::opType() { return ace::OpType_Range; }

ace::OpParameter RangeOnnx::type() { return ace::OpParameter_NONE; }

void RangeOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(RangeOnnx, Range);

}  // namespace converter
}  // namespace ace