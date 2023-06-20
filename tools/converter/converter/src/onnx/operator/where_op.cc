#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(WhereOnnx);

ace::OpType WhereOnnx::opType() { return ace::OpType_Select; }

ace::OpParameter WhereOnnx::type() { return ace::OpParameter_NONE; }

void WhereOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                    OnnxScope *scope) {
  return;
}

REGISTER_CONVERTER(WhereOnnx, Where);

}  // namespace converter
}  // namespace ace