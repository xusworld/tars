#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(DetOnnx);

ace::OpType DetOnnx::opType() { return ace::OpType_Det; }
ace::OpParameter DetOnnx::type() { return ace::OpParameter_NONE; }

void DetOnnx::run(ace::OpT *dstOp, const onnx::NodeProto *onnxNode,
                  OnnxScope *scope) {}

REGISTER_CONVERTER(DetOnnx, Det);

}  // namespace converter
}  // namespace ace