#include <stdio.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(TanhOnnx);

ace::OpType TanhOnnx::opType() { return ace::OpType_TanH; }
ace::OpParameter TanhOnnx::type() { return ace::OpParameter_NONE; }

void TanhOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   OnnxScope* scope) {
  dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TanhOnnx, Tanh);

}  // namespace converter
}  // namespace ace