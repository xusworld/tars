#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(ReshapeOnnx);

ace::OpType ReshapeOnnx::opType() { return ace::OpType_Reshape; }
ace::OpParameter ReshapeOnnx::type() { return ace::OpParameter_Reshape; }

void ReshapeOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
  auto para = new ace::ReshapeT;
  para->dimType = ace::DataFormat_NCHW;
  dstOp->main.value = para;
}

REGISTER_CONVERTER(ReshapeOnnx, Reshape);

}  // namespace converter
}  // namespace ace