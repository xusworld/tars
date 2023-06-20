#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(SigmoidOnnx);

ace::OpType SigmoidOnnx::opType() { return ace::OpType_Sigmoid; }

ace::OpParameter SigmoidOnnx::type() { return ace::OpParameter_NONE; }

void SigmoidOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                      OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(SigmoidOnnx, Sigmoid);

}  // namespace converter
}  // namespace ace