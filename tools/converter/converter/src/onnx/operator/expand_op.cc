#include <glog/logging.h>

#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ExpandOnnx);

ace::OpType ExpandOnnx::opType() { return ace::OpType_BroadcastTo; }

ace::OpParameter ExpandOnnx::type() { return ace::OpParameter_NONE; }

void ExpandOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                     OnnxScope *scope) {
  DCHECK(2 == onnxNode->input_size()) << "ONNX Expand should have 2 inputs!";
  return;
}

REGISTER_CONVERTER(ExpandOnnx, Expand);

}  // namespace converter
}  // namespace ace