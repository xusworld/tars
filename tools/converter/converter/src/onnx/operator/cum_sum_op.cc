#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(CumSumOnnx);

ace::OpType CumSumOnnx::opType() { return ace::OpType_CumSum; }
ace::OpParameter CumSumOnnx::type() { return ace::OpParameter_CumSum; }

void CumSumOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     OnnxScope* scope) {
  auto param = new ace::CumSumT;
  param->exclusive = param->reverse = false;
  for (const auto& attr : onnxNode->attribute()) {
    if (attr.name() == "exclusive") {
      param->exclusive = attr.i();
    } else if (attr.name() == "reverse") {
      param->reverse = attr.i();
    }
  }
  dstOp->main.value = param;
}

REGISTER_CONVERTER(CumSumOnnx, CumSum);

}  // namespace converter
}  // namespace ace