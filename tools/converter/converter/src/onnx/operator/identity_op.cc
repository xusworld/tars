#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(IdentityOnnx);

ace::OpType IdentityOnnx::opType() { return ace::OpType_Identity; }
ace::OpParameter IdentityOnnx::type() { return ace::OpParameter_NONE; }

void IdentityOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                       OnnxScope *scope) {
  // Do nothing
  return;
}

REGISTER_CONVERTER(IdentityOnnx, Identity);

}  // namespace converter
}  // namespace ace