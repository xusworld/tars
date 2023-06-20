#include "ace/schema/ace_generated.h"
#include "src/onnx/onnx_op_converter.h"
#include "src/onnx/onnx_op_converter_register.h"
#include "src/onnx/onnx_scope.h"

namespace ace {
namespace converter {

DECLARE_OP_CONVERTER(SoftmaxOnnx);

ace::OpType SoftmaxOnnx::opType() { return ace::OpType_Softmax; }
ace::OpParameter SoftmaxOnnx::type() { return ace::OpParameter_Axis; }

void SoftmaxOnnx::run(ace::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      OnnxScope* scope) {
  auto axis = new ace::AxisT;
  axis->axis = -1;
  const auto attrSize = onnxNode->attribute_size();
  for (int i = 0; i < attrSize; ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName = attributeProto.name();
    if (attributeName == "axis") {
      axis->axis = (int)attributeProto.i();
    }
  }
  dstOp->main.value = axis;
}

REGISTER_CONVERTER(SoftmaxOnnx, Softmax);

}  // namespace converter
}  // namespace ace